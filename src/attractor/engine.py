"""Pipeline Execution Engine (Section 3 & Section 5 of attractor-spec.md)."""

from __future__ import annotations

import json
import random
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .backends import CallbackBackend, CodergenBackend
from .conditions import evaluate_condition
from .handlers import Handler, HandlerRegistry, ParallelHandler, write_status_file
from .interviewer import Interviewer, normalize_label
from .models import (
    SHAPE_TO_HANDLER_TYPE,
    Checkpoint,
    Context,
    Edge,
    Graph,
    Node,
    Outcome,
    PipelineEvent,
    StageStatus,
)
from .parser import parse_dot
from .transforms import (
    StylesheetApplicationTransform,
    Transform,
    VariableExpansionTransform,
)
from .validator import (
    find_start_nodes,
    find_terminal_nodes,
    validate,
    validate_or_raise,
)


@dataclass
class BackoffConfig:
    initial_delay_ms: int = 10
    backoff_factor: float = 2.0
    max_delay_ms: int = 1000
    jitter: bool = False

    def delay_for_attempt(self, attempt: int) -> float:
        delay_ms = self.initial_delay_ms * (self.backoff_factor ** max(0, attempt - 1))
        delay_ms = min(delay_ms, self.max_delay_ms)
        if self.jitter:
            delay_ms *= random.uniform(0.5, 1.5)
        return delay_ms / 1000.0


@dataclass
class RetryPolicy:
    max_attempts: int = 1
    backoff: BackoffConfig = field(default_factory=BackoffConfig)


def select_edge(
    node: Node, outcome: Outcome, context: Context, graph: Graph
) -> Edge | None:
    """5-step deterministic edge selection algorithm from Section 3.3."""
    edges = graph.outgoing_edges(node.id)
    if not edges:
        return None

    def _best_by_weight_then_lexical(candidates: list[Edge]) -> Edge:
        return sorted(candidates, key=lambda e: (-e.weight, e.to_node))[0]

    # Step 1: Condition-matching edges
    condition_matched: list[Edge] = []
    for edge in edges:
        if edge.condition.strip():
            if evaluate_condition(edge.condition, outcome, context):
                condition_matched.append(edge)
    if condition_matched:
        return _best_by_weight_then_lexical(condition_matched)

    unconditional = [e for e in edges if not e.condition.strip()]

    # Step 2: Preferred label match
    if outcome.preferred_label:
        target_norm = normalize_label(outcome.preferred_label)
        for edge in unconditional:
            if edge.label and normalize_label(edge.label) == target_norm:
                return edge

    # Step 3: Suggested next IDs
    if outcome.suggested_next_ids:
        for suggested_id in outcome.suggested_next_ids:
            for edge in unconditional:
                if edge.to_node == suggested_id:
                    return edge

    # Step 4 & 5: Highest weight with lexical tiebreak on target node ID
    if unconditional:
        return _best_by_weight_then_lexical(unconditional)

    return None


def check_goal_gates(
    graph: Graph, node_outcomes: dict[str, Outcome]
) -> tuple[bool, Node | None]:
    """Verify all visited goal_gate=true nodes achieved SUCCESS or PARTIAL_SUCCESS."""
    for node_id, outcome in node_outcomes.items():
        node = graph.nodes.get(node_id)
        if node and node.goal_gate:
            if outcome.status not in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
                return False, node
    return True, None


def get_retry_target(node: Node, graph: Graph) -> str | None:
    for candidate in (
        node.retry_target,
        node.fallback_retry_target,
        graph.retry_target,
        graph.fallback_retry_target,
    ):
        if candidate and candidate in graph.nodes:
            return candidate
    return None


class ArtifactStore:
    """Named, typed storage for stage outputs (Section 5.5)."""

    FILE_BACKING_THRESHOLD = 100 * 1024

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir
        self._artifacts: dict[str, tuple[dict[str, Any], Any]] = {}
        self._lock = threading.RLock()

    def store(self, artifact_id: str, name: str, data: Any) -> dict[str, Any]:
        serialized = json.dumps(data, default=str)
        size = len(serialized.encode("utf-8"))
        is_file_backed = size > self.FILE_BACKING_THRESHOLD and self.base_dir is not None
        if is_file_backed and self.base_dir is not None:
            art_dir = self.base_dir / "artifacts"
            art_dir.mkdir(parents=True, exist_ok=True)
            fpath = art_dir / f"{artifact_id}.json"
            fpath.write_text(serialized, encoding="utf-8")
            stored: Any = str(fpath)
        else:
            stored = data
        info = {
            "id": artifact_id,
            "name": name,
            "size_bytes": size,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "is_file_backed": is_file_backed,
        }
        with self._lock:
            self._artifacts[artifact_id] = (info, stored)
        return info

    def retrieve(self, artifact_id: str) -> Any:
        with self._lock:
            if artifact_id not in self._artifacts:
                raise KeyError(f"Artifact not found: {artifact_id}")
            info, stored = self._artifacts[artifact_id]
            if info["is_file_backed"]:
                return json.loads(Path(stored).read_text(encoding="utf-8"))
            return stored


class PipelineRunner:
    """Executes Attractor DOT graphs with transforms, validation, checkpoints, and events."""

    def __init__(
        self,
        backend: CodergenBackend | Callable[[Node, str, Context], str | Outcome] | None = None,
        interviewer: Interviewer | None = None,
        logs_root: str | Path | None = None,
        on_event: Callable[[PipelineEvent], None] | None = None,
        backoff: BackoffConfig | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        if callable(backend) and not hasattr(backend, "run"):
            resolved_backend: CodergenBackend | None = CallbackBackend(backend)
        else:
            resolved_backend = backend  # type: ignore[assignment]

        self.registry = HandlerRegistry(backend=resolved_backend, interviewer=interviewer)
        self.logs_root = Path(logs_root) if logs_root else Path(tempfile.mkdtemp(prefix="attractor_run_"))
        self.workspace = Path(workspace).resolve() if workspace else None
        self.on_event = on_event
        self.backoff = backoff or BackoffConfig()
        self.custom_transforms: list[Transform] = []
        self._cancel_event = threading.Event()

        # Wire parallel handler to execute individual branch nodes via registry
        par_handler = self.registry.handlers.get("parallel")
        if isinstance(par_handler, ParallelHandler):
            par_handler.branch_executor = self._execute_single_node_for_parallel

    def register_transform(self, transform: Transform) -> None:
        self.custom_transforms.append(transform)

    def register_handler(self, type_string: str, handler: Handler) -> None:
        self.registry.register(type_string, handler)

    def cancel(self) -> None:
        self._cancel_event.set()

    def _emit(
        self,
        event_type: str,
        pipeline_id: str = "",
        node_id: str | None = None,
        **data: Any,
    ) -> None:
        if self.on_event:
            self.on_event(
                PipelineEvent(
                    type=event_type,
                    pipeline_id=pipeline_id,
                    node_id=node_id,
                    data=data,
                )
            )

    def prepare(self, dot_source_or_graph: str | Graph) -> tuple[Graph, list[Any]]:
        graph = (
            parse_dot(dot_source_or_graph)
            if isinstance(dot_source_or_graph, str)
            else dot_source_or_graph.model_copy(deep=True)
        )
        builtin_transforms: list[Transform] = [
            VariableExpansionTransform(),
            StylesheetApplicationTransform(),
        ]
        for t in builtin_transforms + self.custom_transforms:
            graph = t.apply(graph)
        diagnostics = validate(graph, known_types=self.registry.known_types())
        return graph, diagnostics

    def _execute_single_node_for_parallel(
        self, node_id: str, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        node = graph.nodes[node_id]
        handler = self.registry.resolve(node)
        return handler.execute(node, context, graph, logs_root)

    def run(
        self,
        dot_source_or_graph: str | Graph,
        context: Context | None = None,
        resume_from_checkpoint: Checkpoint | None = None,
        pipeline_id: str = "run-1",
    ) -> Outcome:
        graph, _ = self.prepare(dot_source_or_graph)
        validate_or_raise(graph, known_types=self.registry.known_types())

        self.logs_root.mkdir(parents=True, exist_ok=True)
        ctx = context if context is not None else Context()

        for k, v in graph.attrs.items():
            ctx.set(f"graph.{k}", v)
        ctx.set("graph.goal", graph.goal)
        if self.workspace and not ctx.get("workspace"):
            ctx.set("workspace", str(self.workspace))

        manifest = {
            "pipeline_id": pipeline_id,
            "name": graph.name,
            "goal": graph.goal,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.logs_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        completed_nodes: list[str] = []
        node_outcomes: dict[str, Outcome] = {}
        node_retries: dict[str, int] = {}

        starts = find_start_nodes(graph)
        terminals = {n.id for n in find_terminal_nodes(graph)}
        current_node = starts[0]

        if resume_from_checkpoint is not None:
            ctx.apply_updates(resume_from_checkpoint.context)
            completed_nodes = list(resume_from_checkpoint.completed_nodes)
            node_retries = dict(resume_from_checkpoint.node_retries)
            for nid in completed_nodes:
                status_file = self.logs_root / nid / "status.json"
                if status_file.exists():
                    sdata = json.loads(status_file.read_text(encoding="utf-8"))
                    node_outcomes[nid] = Outcome(
                        status=StageStatus(sdata.get("outcome", "success")),
                        preferred_label=sdata.get("preferred_label", ""),
                        suggested_next_ids=sdata.get("suggested_next_ids", []),
                        context_updates=sdata.get("context_updates", {}),
                        notes=sdata.get("notes", ""),
                    )
                else:
                    node_outcomes[nid] = Outcome(status=StageStatus.SUCCESS)

            last_id = resume_from_checkpoint.current_node
            if last_id in terminals:
                return Outcome(
                    status=StageStatus.SUCCESS,
                    notes="Resumed at terminal node",
                    completed_nodes=completed_nodes,
                )
            if last_id in graph.nodes:
                last_node = graph.nodes[last_id]
                last_outcome = node_outcomes.get(
                    last_id, Outcome(status=StageStatus.SUCCESS)
                )
                nxt = select_edge(last_node, last_outcome, ctx, graph)
                if nxt:
                    current_node = graph.nodes[nxt.to_node]

        self._emit("PipelineStarted", pipeline_id=pipeline_id, name=graph.name, goal=graph.goal)
        start_time = time.monotonic()
        step_guard = 0
        max_steps = 500

        while step_guard < max_steps:
            step_guard += 1
            if self._cancel_event.is_set():
                self._emit("PipelineFailed", pipeline_id=pipeline_id, error="Pipeline cancelled")
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason="Pipeline cancelled",
                    completed_nodes=completed_nodes,
                )

            node = graph.nodes[current_node.id]
            ctx.set("current_node", node.id)

            # Step 1: Check for terminal node
            if node.id in terminals:
                gate_ok, failed_gate = check_goal_gates(graph, node_outcomes)
                if not gate_ok and failed_gate is not None:
                    retry_target = get_retry_target(failed_gate, graph)
                    if retry_target:
                        current_node = graph.nodes[retry_target]
                        continue
                    self._emit(
                        "PipelineFailed",
                        pipeline_id=pipeline_id,
                        node_id=node.id,
                        error=f"Goal gate '{failed_gate.id}' unsatisfied and no retry target",
                    )
                    return Outcome(
                        status=StageStatus.FAIL,
                        failure_reason=f"Goal gate '{failed_gate.id}' unsatisfied and no retry target",
                        completed_nodes=completed_nodes,
                    )

                # Execute exit handler and save final checkpoint
                exit_handler = self.registry.resolve(node)
                exit_outcome = exit_handler.execute(node, ctx, graph, self.logs_root)
                completed_nodes.append(node.id)
                node_outcomes[node.id] = exit_outcome
                cp = Checkpoint(
                    current_node=node.id,
                    completed_nodes=completed_nodes,
                    node_retries=node_retries,
                    context=ctx.snapshot(),
                    logs=ctx.logs,
                )
                cp.save(self.logs_root / "checkpoint.json")
                self._emit("CheckpointSaved", pipeline_id=pipeline_id, node_id=node.id)
                duration = round(time.monotonic() - start_time, 3)
                self._emit(
                    "PipelineCompleted",
                    pipeline_id=pipeline_id,
                    node_id=node.id,
                    duration=duration,
                )
                return Outcome(
                    status=StageStatus.SUCCESS,
                    notes="Pipeline completed",
                    completed_nodes=completed_nodes,
                )

            # Step 2: Execute node handler with retry policy
            self._emit(
                "StageStarted",
                pipeline_id=pipeline_id,
                node_id=node.id,
                label=node.label,
                shape=node.shape,
            )
            stage_t0 = time.monotonic()
            max_retries = (
                node.max_retries
                if node.max_retries is not None
                else graph.default_max_retries
            )
            policy = RetryPolicy(max_attempts=max(1, max_retries + 1), backoff=self.backoff)
            outcome = self._execute_with_retry(
                node, ctx, graph, policy, node_retries, pipeline_id
            )
            stage_dur = round(time.monotonic() - stage_t0, 3)

            # Auto-status check (Appendix C)
            stage_dir = self.logs_root / node.id
            status_path = stage_dir / "status.json"
            if node.auto_status and not status_path.exists():
                outcome = Outcome(
                    status=StageStatus.SUCCESS,
                    notes="auto-status: handler completed without writing status",
                )
                write_status_file(stage_dir, outcome)
            elif node.id not in starts and not status_path.exists():
                write_status_file(stage_dir, outcome)

            # Step 3: Record completion
            completed_nodes.append(node.id)
            node_outcomes[node.id] = outcome

            if outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
                self._emit(
                    "StageCompleted",
                    pipeline_id=pipeline_id,
                    node_id=node.id,
                    label=node.label,
                    duration=stage_dur,
                    outcome=outcome.status.value,
                    notes=outcome.notes,
                )
            else:
                self._emit(
                    "StageFailed",
                    pipeline_id=pipeline_id,
                    node_id=node.id,
                    label=node.label,
                    duration=stage_dur,
                    error=outcome.failure_reason or outcome.notes,
                )

            # Step 4: Apply context updates from outcome
            if outcome.context_updates:
                ctx.apply_updates(outcome.context_updates)
            ctx.set("outcome", outcome.status.value)
            if outcome.preferred_label:
                ctx.set("preferred_label", outcome.preferred_label)

            # Step 5: Save checkpoint
            cp = Checkpoint(
                current_node=node.id,
                completed_nodes=completed_nodes,
                node_retries=node_retries,
                context=ctx.snapshot(),
                logs=ctx.logs,
            )
            cp.save(self.logs_root / "checkpoint.json")
            self._emit("CheckpointSaved", pipeline_id=pipeline_id, node_id=node.id)

            # Step 6: Select next edge (including failure routing from Section 3.7)
            next_edge = select_edge(node, outcome, ctx, graph)

            if outcome.status == StageStatus.FAIL:
                # Check if selected edge was explicitly matching outcome=fail
                has_explicit_fail_edge = bool(
                    next_edge
                    and next_edge.condition
                    and "fail" in next_edge.condition
                )
                if not has_explicit_fail_edge:
                    jump_id = node.retry_target or node.fallback_retry_target
                    if jump_id and jump_id in graph.nodes:
                        current_node = graph.nodes[jump_id]
                        continue
                    if next_edge is None:
                        self._emit(
                            "PipelineFailed",
                            pipeline_id=pipeline_id,
                            node_id=node.id,
                            error=outcome.failure_reason or "Stage failed",
                        )
                        outcome.completed_nodes = completed_nodes
                        return outcome

            if next_edge is None:
                if outcome.status == StageStatus.FAIL:
                    outcome.completed_nodes = completed_nodes
                    return outcome
                return Outcome(
                    status=StageStatus.SUCCESS,
                    notes="Pipeline completed",
                    completed_nodes=completed_nodes,
                )

            # Step 7: Advance to next node (if parallel fan-out just ran, jump to fan-in if direct)
            resolved_type = node.node_type or SHAPE_TO_HANDLER_TYPE.get(node.shape, "codergen")
            if resolved_type == "parallel":
                # Parallel handler already executed immediate branch nodes; record them and advance to their convergence target
                branch_targets = [e.to_node for e in graph.outgoing_edges(node.id)]
                fan_in_candidates: list[str] = []
                for bt in branch_targets:
                    if bt not in completed_nodes:
                        completed_nodes.append(bt)
                        node_outcomes[bt] = Outcome(status=StageStatus.SUCCESS)
                    for out_e in graph.outgoing_edges(bt):
                        if out_e.to_node not in fan_in_candidates:
                            fan_in_candidates.append(out_e.to_node)
                if fan_in_candidates:
                    current_node = graph.nodes[fan_in_candidates[0]]
                    continue

            current_node = graph.nodes[next_edge.to_node]

        return Outcome(
            status=StageStatus.FAIL,
            failure_reason="Max pipeline traversal steps exceeded",
            completed_nodes=completed_nodes,
        )

    def _execute_with_retry(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        retry_policy: RetryPolicy,
        node_retries: dict[str, int],
        pipeline_id: str,
    ) -> Outcome:
        handler = self.registry.resolve(node)
        for attempt in range(1, retry_policy.max_attempts + 1):
            try:
                outcome = handler.execute(node, context, graph, self.logs_root)
            except Exception as exc:
                if attempt < retry_policy.max_attempts:
                    node_retries[node.id] = node_retries.get(node.id, 0) + 1
                    context.set(f"internal.retry_count.{node.id}", node_retries[node.id])
                    delay = retry_policy.backoff.delay_for_attempt(attempt)
                    self._emit(
                        "StageRetrying",
                        pipeline_id=pipeline_id,
                        node_id=node.id,
                        attempt=attempt,
                        delay=delay,
                    )
                    time.sleep(delay)
                    continue
                return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))

            if outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
                node_retries[node.id] = 0
                context.set(f"internal.retry_count.{node.id}", 0)
                return outcome

            if outcome.status in (StageStatus.RETRY, StageStatus.FAIL):
                if attempt < retry_policy.max_attempts:
                    node_retries[node.id] = node_retries.get(node.id, 0) + 1
                    context.set(f"internal.retry_count.{node.id}", node_retries[node.id])
                    delay = retry_policy.backoff.delay_for_attempt(attempt)
                    self._emit(
                        "StageRetrying",
                        pipeline_id=pipeline_id,
                        node_id=node.id,
                        attempt=attempt,
                        delay=delay,
                    )
                    time.sleep(delay)
                    continue
                if node.allow_partial:
                    return Outcome(
                        status=StageStatus.PARTIAL_SUCCESS,
                        notes="retries exhausted, partial accepted",
                    )
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=outcome.failure_reason or "max retries exceeded",
                )

        return Outcome(status=StageStatus.FAIL, failure_reason="max retries exceeded")


def run_pipeline(
    dot_source_or_graph: str | Graph,
    context: Context | None = None,
    llm_callback: Callable[[Node, str, Context], str | Outcome] | None = None,
    interviewer: Interviewer | None = None,
    logs_root: str | Path | None = None,
    workspace: str | Path | None = None,
) -> Outcome:
    """Convenience function to run an Attractor pipeline end-to-end."""
    runner = PipelineRunner(
        backend=llm_callback,
        interviewer=interviewer,
        logs_root=logs_root,
        workspace=workspace,
    )
    return runner.run(dot_source_or_graph, context=context)
