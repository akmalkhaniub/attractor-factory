"""FastAPI HTTP Server Mode for Attractor (Section 9.5 of attractor-spec.md)."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .backends import UnifiedLLMBackend
from .engine import PipelineRunner
from .interviewer import AutoApproveInterviewer, WebInteractiveInterviewer
from .models import (
    Graph,
    Outcome,
    PipelineEvent,
    Question,
    Severity,
    StageStatus,
)
from .presets import PRESET_PIPELINES
from .validator import ValidationError


class PipelineCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dot_source: str = Field(..., min_length=10)
    auto_approve: bool = Field(default=False)
    step_delay_ms: int = Field(default=350, ge=0, le=5000)
    workspace: str | None = Field(default=None)


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dot_source: str = Field(..., min_length=1)


class AnswerSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    value: str = Field(..., min_length=1)
    notes: str = Field(default="")


class PipelineRunRecord:
    def __init__(
        self,
        pipeline_id: str,
        graph: Graph,
        dot_source: str,
        logs_root: Path,
        auto_approve: bool,
        step_delay_ms: int,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.graph = graph
        self.dot_source = dot_source
        self.logs_root = logs_root
        self.auto_approve = auto_approve
        self.step_delay_ms = step_delay_ms
        self.status: str = "running"
        self.current_node: str | None = None
        self.completed_nodes: list[str] = []
        self.node_statuses: dict[str, str] = {}
        self.stage_artifacts: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.outcome: Outcome | None = None
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None
        self._lock = threading.Lock()
        self.web_interviewer = WebInteractiveInterviewer(
            on_question=self._on_question_asked
        )
        self.runner: PipelineRunner | None = None

    def _on_question_asked(self, q: Question) -> None:
        with self._lock:
            self.status = "awaiting_human_approval"
            self.events.append(
                PipelineEvent(
                    type="InterviewStarted",
                    pipeline_id=self.pipeline_id,
                    node_id=q.stage,
                    data={"question": q.model_dump()},
                ).model_dump()
            )

    def on_event(self, evt: PipelineEvent) -> None:
        if self.step_delay_ms > 0 and evt.type == "StageStarted":
            time.sleep(self.step_delay_ms / 1000.0)

        with self._lock:
            if evt.type == "StageStarted" and evt.node_id:
                self.current_node = evt.node_id
                self.node_statuses[evt.node_id] = "running"
            elif evt.type == "StageCompleted" and evt.node_id:
                self.completed_nodes.append(evt.node_id)
                self.node_statuses[evt.node_id] = evt.data.get("outcome", "success")
                self._collect_stage_artifacts(evt.node_id)
            elif evt.type == "StageFailed" and evt.node_id:
                self.node_statuses[evt.node_id] = "fail"
                self._collect_stage_artifacts(evt.node_id)
            elif evt.type == "PipelineCompleted":
                if evt.node_id:
                    self.completed_nodes.append(evt.node_id)
                    self.node_statuses[evt.node_id] = "success"
                self.current_node = None
                self.status = "completed"
                self.completed_at = datetime.now(timezone.utc).isoformat()
            elif evt.type == "PipelineFailed":
                self.current_node = None
                self.status = "failed"
                self.completed_at = datetime.now(timezone.utc).isoformat()

            self.events.append(evt.model_dump())

    def _collect_stage_artifacts(self, node_id: str) -> None:
        stage_dir = self.logs_root / node_id
        if not stage_dir.exists():
            return
        art: dict[str, Any] = {}
        prompt_file = stage_dir / "prompt.md"
        resp_file = stage_dir / "response.md"
        status_file = stage_dir / "status.json"
        if prompt_file.exists():
            art["prompt"] = prompt_file.read_text(encoding="utf-8")
        if resp_file.exists():
            art["response"] = resp_file.read_text(encoding="utf-8")
        if status_file.exists():
            try:
                art["status"] = json.loads(status_file.read_text(encoding="utf-8"))
            except Exception:
                art["status"] = {}
        self.stage_artifacts[node_id] = art

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.graph.name,
            "label": self.graph.label,
            "goal": self.graph.goal,
            "status": self.status,
            "current_node": self.current_node,
            "completed_nodes": self.completed_nodes,
            "node_statuses": self.node_statuses,
            "pending_questions": [
                q.model_dump() for q in self.web_interviewer.get_pending_questions()
            ],
            "stage_artifacts": self.stage_artifacts,
            "events": self.events[-50:],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "graph": {
                "name": self.graph.name,
                "goal": self.graph.goal,
                "nodes": [
                    {
                        "id": n.id,
                        "label": n.label,
                        "shape": n.shape,
                        "type": n.node_type,
                        "prompt": n.prompt,
                        "goal_gate": n.goal_gate,
                        "classes": n.classes,
                    }
                    for n in self.graph.nodes.values()
                ],
                "edges": [
                    {
                        "from": e.from_node,
                        "to": e.to_node,
                        "label": e.label,
                        "condition": e.condition,
                        "weight": e.weight,
                    }
                    for e in self.graph.edges
                ],
            },
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runs: dict[str, PipelineRunRecord] = {}
    app.state.base_logs_dir = Path(tempfile.mkdtemp(prefix="attractor_server_"))
    yield


app = FastAPI(
    title="Attractor Software Factory Console",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/presets")
async def get_presets() -> dict[str, Any]:
    return {"presets": PRESET_PIPELINES}


@app.post("/api/pipelines/validate")
async def validate_pipeline(payload: ValidateRequest) -> dict[str, Any]:
    runner = PipelineRunner()
    try:
        graph, diagnostics = runner.prepare(payload.dot_source)
    except Exception as exc:
        return {
            "valid": False,
            "diagnostics": [
                {
                    "rule": "dot_syntax",
                    "severity": "error",
                    "message": str(exc),
                    "node_id": None,
                    "edge": None,
                    "fix": "Check DOT syntax for unbalanced brackets or invalid tokens.",
                }
            ],
            "graph": None,
        }

    has_errors = any(d.severity == Severity.ERROR for d in diagnostics)
    return {
        "valid": not has_errors,
        "diagnostics": [d.model_dump() for d in diagnostics],
        "graph": {
            "name": graph.name,
            "goal": graph.goal,
            "label": graph.label,
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "shape": n.shape,
                    "type": n.node_type,
                    "prompt": n.prompt,
                    "goal_gate": n.goal_gate,
                    "classes": n.classes,
                }
                for n in graph.nodes.values()
            ],
            "edges": [
                {
                    "from": e.from_node,
                    "to": e.to_node,
                    "label": e.label,
                    "condition": e.condition,
                    "weight": e.weight,
                }
                for e in graph.edges
            ],
        },
    }


@app.post("/pipelines", status_code=status.HTTP_202_ACCEPTED)
@app.post("/api/pipelines", status_code=status.HTTP_202_ACCEPTED)
async def start_pipeline(payload: PipelineCreateRequest, request: Request) -> dict[str, Any]:
    probe_runner = PipelineRunner()
    try:
        graph, diagnostics = probe_runner.prepare(payload.dot_source)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"DOT parse error: {exc}",
        ) from exc

    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Pipeline has validation errors",
                "diagnostics": [d.model_dump() for d in errors],
            },
        )

    pipeline_id = f"run-{uuid.uuid4().hex[:8]}"
    logs_root: Path = request.app.state.base_logs_dir / pipeline_id
    record = PipelineRunRecord(
        pipeline_id=pipeline_id,
        graph=graph,
        dot_source=payload.dot_source,
        logs_root=logs_root,
        auto_approve=payload.auto_approve,
        step_delay_ms=payload.step_delay_ms,
    )
    interviewer = (
        AutoApproveInterviewer() if payload.auto_approve else record.web_interviewer
    )
    runner = PipelineRunner(
        backend=UnifiedLLMBackend(),
        interviewer=interviewer,
        logs_root=logs_root,
        workspace=payload.workspace,
        on_event=record.on_event,
    )
    record.runner = runner
    request.app.state.runs[pipeline_id] = record

    def _bg_worker() -> None:
        try:
            out = runner.run(payload.dot_source, pipeline_id=pipeline_id)
            record.outcome = out
            if record.status not in ("completed", "failed"):
                record.status = (
                    "completed"
                    if out.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)
                    else "failed"
                )
        except ValidationError as vexc:
            record.status = "failed"
            record.on_event(
                PipelineEvent(
                    type="PipelineFailed",
                    pipeline_id=pipeline_id,
                    data={"error": str(vexc)},
                )
            )
        except Exception as exc:
            record.status = "failed"
            record.on_event(
                PipelineEvent(
                    type="PipelineFailed",
                    pipeline_id=pipeline_id,
                    data={"error": str(exc)},
                )
            )

    thread = threading.Thread(target=_bg_worker, daemon=True)
    thread.start()
    return record.to_summary_dict()


@app.get("/pipelines/{pipeline_id}")
@app.get("/api/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str, request: Request) -> dict[str, Any]:
    record: PipelineRunRecord | None = request.app.state.runs.get(pipeline_id)
    if not record:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return record.to_summary_dict()


@app.get("/pipelines/{pipeline_id}/events")
@app.get("/api/pipelines/{pipeline_id}/events")
async def stream_pipeline_events(pipeline_id: str, request: Request) -> StreamingResponse:
    record: PipelineRunRecord | None = request.app.state.runs.get(pipeline_id)
    if not record:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    async def _event_generator():
        cursor = 0
        while True:
            while cursor < len(record.events):
                evt = record.events[cursor]
                cursor += 1
                yield f"data: {json.dumps(evt)}\n\n"
            if record.status in ("completed", "failed") and cursor >= len(record.events):
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.get("/pipelines/{pipeline_id}/questions")
@app.get("/api/pipelines/{pipeline_id}/questions")
async def get_pending_questions(pipeline_id: str, request: Request) -> dict[str, Any]:
    record: PipelineRunRecord | None = request.app.state.runs.get(pipeline_id)
    if not record:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return {
        "questions": [
            q.model_dump() for q in record.web_interviewer.get_pending_questions()
        ]
    }


@app.post("/pipelines/{pipeline_id}/questions/{question_id}/answer")
@app.post("/api/pipelines/{pipeline_id}/questions/{question_id}/answer")
async def answer_question(
    pipeline_id: str,
    question_id: str,
    payload: AnswerSubmitRequest,
    request: Request,
) -> dict[str, Any]:
    record: PipelineRunRecord | None = request.app.state.runs.get(pipeline_id)
    if not record:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    ok = record.web_interviewer.submit_answer(
        question_id=question_id, value=payload.value, text=payload.notes
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Pending question not found")
    record.status = "running"
    record.events.append(
        PipelineEvent(
            type="InterviewCompleted",
            pipeline_id=pipeline_id,
            data={"question_id": question_id, "value": payload.value},
        ).model_dump()
    )
    return {"status": "resumed", "pipeline_id": pipeline_id, "selected": payload.value}


@app.post("/pipelines/{pipeline_id}/cancel")
@app.post("/api/pipelines/{pipeline_id}/cancel")
async def cancel_pipeline(pipeline_id: str, request: Request) -> dict[str, Any]:
    record: PipelineRunRecord | None = request.app.state.runs.get(pipeline_id)
    if not record:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    if record.runner:
        record.runner.cancel()
    record.status = "failed"
    return {"status": "cancelled", "pipeline_id": pipeline_id}


@app.get("/pipelines/{pipeline_id}/checkpoint")
@app.get("/api/pipelines/{pipeline_id}/checkpoint")
async def get_checkpoint(pipeline_id: str, request: Request) -> dict[str, Any]:
    record: PipelineRunRecord | None = request.app.state.runs.get(pipeline_id)
    if not record:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    cp_path = record.logs_root / "checkpoint.json"
    if not cp_path.exists():
        return {"checkpoint": None}
    return {"checkpoint": json.loads(cp_path.read_text(encoding="utf-8"))}


@app.get("/pipelines/{pipeline_id}/context")
@app.get("/api/pipelines/{pipeline_id}/context")
async def get_context(pipeline_id: str, request: Request) -> dict[str, Any]:
    record: PipelineRunRecord | None = request.app.state.runs.get(pipeline_id)
    if not record:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    cp_path = record.logs_root / "checkpoint.json"
    if not cp_path.exists():
        return {"context": {"graph.goal": record.graph.goal}}
    data = json.loads(cp_path.read_text(encoding="utf-8"))
    return {"context": data.get("context", {})}


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_console() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
