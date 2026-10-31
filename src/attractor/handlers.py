"""Node Handlers & Handler Registry (Section 4 of attractor-spec.md)."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Protocol

from .backends import CodergenBackend, UnifiedLLMBackend
from .interviewer import (
    AutoApproveInterviewer,
    Interviewer,
    normalize_label,
    parse_accelerator_key,
)
from .models import (
    SHAPE_TO_HANDLER_TYPE,
    AnswerValue,
    Context,
    Graph,
    Node,
    Option,
    Outcome,
    Question,
    QuestionType,
    StageStatus,
)


class Handler(Protocol):
    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome: ...


def write_status_file(stage_dir: Path, outcome: Outcome) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    status_path = stage_dir / "status.json"
    status_path.write_text(
        json.dumps(outcome.to_status_dict(), indent=2, default=str),
        encoding="utf-8",
    )


class StartHandler:
    """No-op handler for pipeline entry point (shape=Mdiamond)."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes="Pipeline started")


class ExitHandler:
    """No-op handler for pipeline exit point (shape=Msquare)."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes="Pipeline exit reached")


class CodergenHandler:
    """Default LLM task handler (shape=box, type=codergen)."""

    def __init__(self, backend: CodergenBackend | None = None) -> None:
        self.backend = backend

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        prompt = node.prompt if node.prompt else node.label
        prompt = prompt.replace("$goal", graph.goal)

        stage_dir = logs_root / node.id
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        if self.backend is not None:
            try:
                result = self.backend.run(node, prompt, context)
                if isinstance(result, Outcome):
                    response_text = result.notes or f"Outcome: {result.status.value}"
                    (stage_dir / "response.md").write_text(response_text, encoding="utf-8")
                    write_status_file(stage_dir, result)
                    return result
                response_text = str(result)
            except Exception as exc:
                fail_outcome = Outcome(
                    status=StageStatus.FAIL, failure_reason=str(exc)
                )
                write_status_file(stage_dir, fail_outcome)
                return fail_outcome
        else:
            response_text = f"[Simulated] Response for stage: {node.id}"

        (stage_dir / "response.md").write_text(response_text, encoding="utf-8")
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
            context_updates={
                "last_stage": node.id,
                "last_response": response_text[:200],
            },
        )
        write_status_file(stage_dir, outcome)
        return outcome


class WaitForHumanHandler:
    """Blocks pipeline execution until a human selects an outgoing edge choice (shape=hexagon)."""

    def __init__(self, interviewer: Interviewer | None = None) -> None:
        self.interviewer: Interviewer = interviewer or AutoApproveInterviewer()

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        edges = graph.outgoing_edges(node.id)
        if not edges:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No outgoing edges for human gate",
            )

        options: list[Option] = []
        for edge in edges:
            label = edge.label or edge.to_node
            key = parse_accelerator_key(label)
            options.append(Option(key=key, label=label, to_node=edge.to_node))

        question = Question(
            text=node.label or "Select an option:",
            type=QuestionType.MULTIPLE_CHOICE,
            options=options,
            stage=node.id,
        )

        answer = self.interviewer.ask(question)

        if answer.value == AnswerValue.TIMEOUT:
            default_choice = str(node.attrs.get("human.default_choice", ""))
            if default_choice:
                answer.value = default_choice
            else:
                return Outcome(
                    status=StageStatus.RETRY,
                    failure_reason="human gate timeout, no default",
                )

        if answer.value == AnswerValue.SKIPPED:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="human skipped interaction",
            )

        selected: Option | None = answer.selected_option
        if selected is None:
            for opt in options:
                if (
                    opt.key.lower() == answer.value.lower()
                    or opt.label.lower() == answer.value.lower()
                    or normalize_label(opt.label) == normalize_label(answer.value)
                    or opt.to_node.lower() == answer.value.lower()
                ):
                    selected = opt
                    break
        if selected is None:
            selected = options[0]

        stage_dir = logs_root / node.id
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            preferred_label=selected.label,
            suggested_next_ids=[selected.to_node],
            notes=f"Human selected [{selected.key}] {selected.label}",
            context_updates={
                "human.gate.selected": selected.key,
                "human.gate.label": selected.label,
            },
        )
        write_status_file(stage_dir, outcome)
        return outcome


class ConditionalHandler:
    """Pass-through handler for diamond-shaped conditional routing points."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        stage_dir = logs_root / node.id
        # Preserve prior stage's outcome status so outgoing condition="outcome=success" works seamlessly
        prev_status_str = context.get_string("outcome", "success")
        try:
            preserved_status = StageStatus(prev_status_str)
        except ValueError:
            preserved_status = StageStatus.SUCCESS

        outcome = Outcome(
            status=preserved_status,
            preferred_label=context.get_string("preferred_label", ""),
            notes=f"Conditional node evaluated: {node.id}",
        )
        write_status_file(stage_dir, outcome)
        return outcome


class ParallelHandler:
    """Fans out execution to multiple branches concurrently (shape=component)."""

    def __init__(
        self,
        branch_executor: Callable[[str, Context, Graph, Path], Outcome] | None = None,
    ) -> None:
        self.branch_executor = branch_executor

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        branches = graph.outgoing_edges(node.id)
        join_policy = str(node.attrs.get("join_policy", "wait_all"))
        max_parallel = int(node.attrs.get("max_parallel", 4))

        results: list[dict[str, Any]] = []

        def _run_branch(target_id: str) -> dict[str, Any]:
            branch_ctx = context.clone()
            if self.branch_executor:
                out = self.branch_executor(target_id, branch_ctx, graph, logs_root)
            else:
                out = Outcome(
                    status=StageStatus.SUCCESS,
                    notes=f"Branch {target_id} completed",
                )
            return {
                "id": target_id,
                "outcome": str(out.status.value),
                "notes": out.notes,
                "score": 1.0 if out.status == StageStatus.SUCCESS else 0.0,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
            futures = [pool.submit(_run_branch, edge.to_node) for edge in branches]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        # Deterministic ordering matching branch declaration order
        order_map = {edge.to_node: idx for idx, edge in enumerate(branches)}
        results.sort(key=lambda r: order_map.get(r["id"], 999))

        context.set("parallel.results", results)
        success_count = sum(1 for r in results if r["outcome"] == StageStatus.SUCCESS.value)
        fail_count = sum(1 for r in results if r["outcome"] == StageStatus.FAIL.value)

        status = StageStatus.SUCCESS
        if join_policy == "wait_all" and fail_count > 0:
            status = StageStatus.PARTIAL_SUCCESS
        elif join_policy == "first_success":
            status = StageStatus.SUCCESS if success_count > 0 else StageStatus.FAIL

        stage_dir = logs_root / node.id
        outcome = Outcome(
            status=status,
            notes=f"Parallel fan-out completed ({success_count}/{len(results)} succeeded)",
            context_updates={"parallel.results": results},
        )
        write_status_file(stage_dir, outcome)
        return outcome


class FanInHandler:
    """Consolidates results from a preceding parallel node (shape=tripleoctagon)."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        results: list[dict[str, Any]] = context.get("parallel.results") or []
        stage_dir = logs_root / node.id
        if not results:
            fail_out = Outcome(
                status=StageStatus.FAIL,
                failure_reason="No parallel results to evaluate",
            )
            write_status_file(stage_dir, fail_out)
            return fail_out

        rank_map = {
            StageStatus.SUCCESS.value: 0,
            StageStatus.PARTIAL_SUCCESS.value: 1,
            StageStatus.RETRY.value: 2,
            StageStatus.FAIL.value: 3,
        }
        sorted_candidates = sorted(
            results,
            key=lambda c: (
                rank_map.get(c.get("outcome", "fail"), 3),
                -float(c.get("score", 0.0)),
                str(c.get("id", "")),
            ),
        )
        best = sorted_candidates[0]
        if all(c.get("outcome") == StageStatus.FAIL.value for c in results):
            fail_out = Outcome(
                status=StageStatus.FAIL,
                failure_reason="All parallel candidates failed",
            )
            write_status_file(stage_dir, fail_out)
            return fail_out

        outcome = Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Selected best candidate: {best['id']}",
            context_updates={
                "parallel.fan_in.best_id": best["id"],
                "parallel.fan_in.best_outcome": best["outcome"],
            },
        )
        write_status_file(stage_dir, outcome)
        return outcome


class ToolHandler:
    """Executes an external tool or shell command (shape=parallelogram)."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        command = str(
            node.attrs.get("tool_command")
            or node.attrs.get("command")
            or ""
        )
        stage_dir = logs_root / node.id
        stage_dir.mkdir(parents=True, exist_ok=True)

        if not command:
            out = Outcome(
                status=StageStatus.FAIL,
                failure_reason="No tool_command specified",
            )
            write_status_file(stage_dir, out)
            return out

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            combined = stdout if stdout else stderr
            (stage_dir / "response.md").write_text(
                f"```text\n$ {command}\n{combined}\n```", encoding="utf-8"
            )
            if proc.returncode == 0:
                out = Outcome(
                    status=StageStatus.SUCCESS,
                    notes=f"Tool completed: {command}",
                    context_updates={"tool.output": stdout},
                )
            else:
                out = Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=stderr or f"Command exited with code {proc.returncode}",
                    context_updates={"tool.output": combined},
                )
            write_status_file(stage_dir, out)
            return out
        except Exception as exc:
            out = Outcome(status=StageStatus.FAIL, failure_reason=str(exc))
            write_status_file(stage_dir, out)
            return out


class ManagerLoopHandler:
    """Supervisor loop handler (shape=house, type=stack.manager_loop)."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        stage_dir = logs_root / node.id
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Supervisor loop completed for {node.id}",
        )
        write_status_file(stage_dir, outcome)
        return outcome


class HandlerRegistry:
    """Resolves node types and shapes to Handler instances (Section 4.2)."""

    def __init__(
        self,
        backend: CodergenBackend | None = None,
        interviewer: Interviewer | None = None,
    ) -> None:
        self.default_handler: Handler = CodergenHandler(
            backend=backend if backend is not None else UnifiedLLMBackend()
        )
        self.handlers: dict[str, Handler] = {
            "start": StartHandler(),
            "exit": ExitHandler(),
            "codergen": self.default_handler,
            "wait.human": WaitForHumanHandler(interviewer=interviewer),
            "conditional": ConditionalHandler(),
            "parallel": ParallelHandler(),
            "parallel.fan_in": FanInHandler(),
            "tool": ToolHandler(),
            "stack.manager_loop": ManagerLoopHandler(),
        }

    def register(self, type_string: str, handler: Handler) -> None:
        self.handlers[type_string] = handler

    def known_types(self) -> set[str]:
        return set(self.handlers.keys())

    def resolve(self, node: Node) -> Handler:
        if node.node_type and node.node_type in self.handlers:
            return self.handlers[node.node_type]
        handler_type = SHAPE_TO_HANDLER_TYPE.get(node.shape, "")
        if handler_type in self.handlers:
            return self.handlers[handler_type]
        return self.default_handler
