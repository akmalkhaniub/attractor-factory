"""Core data models for the Attractor Software Factory specification."""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StageStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    FAIL = "fail"
    SKIPPED = "skipped"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class QuestionType(StrEnum):
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    FREEFORM = "freeform"
    CONFIRMATION = "confirmation"


class AnswerValue(StrEnum):
    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


SHAPE_TO_HANDLER_TYPE: dict[str, str] = {
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}

VALID_FIDELITY_MODES: set[str] = {
    "full",
    "truncate",
    "compact",
    "summary:low",
    "summary:medium",
    "summary:high",
}


class Node(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    attrs: dict[str, Any] = Field(default_factory=dict)
    classes: list[str] = Field(default_factory=list)
    explicit_attrs: set[str] = Field(default_factory=set)

    @property
    def label(self) -> str:
        return str(self.attrs.get("label", self.id))

    @property
    def shape(self) -> str:
        return str(self.attrs.get("shape", "box"))

    @property
    def node_type(self) -> str:
        return str(self.attrs.get("type", ""))

    @property
    def prompt(self) -> str:
        return str(self.attrs.get("prompt", ""))

    @prompt.setter
    def prompt(self, value: str) -> None:
        self.attrs["prompt"] = value

    @property
    def max_retries(self) -> int | None:
        val = self.attrs.get("max_retries")
        if val is None:
            return None
        return int(val)

    @property
    def goal_gate(self) -> bool:
        val = self.attrs.get("goal_gate", False)
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    @property
    def retry_target(self) -> str:
        return str(self.attrs.get("retry_target", ""))

    @property
    def fallback_retry_target(self) -> str:
        return str(self.attrs.get("fallback_retry_target", ""))

    @property
    def fidelity(self) -> str:
        return str(self.attrs.get("fidelity", ""))

    @property
    def thread_id(self) -> str:
        return str(self.attrs.get("thread_id", ""))

    @property
    def timeout(self) -> str:
        return str(self.attrs.get("timeout", ""))

    @property
    def llm_model(self) -> str:
        return str(self.attrs.get("llm_model", ""))

    @property
    def llm_provider(self) -> str:
        return str(self.attrs.get("llm_provider", ""))

    @property
    def reasoning_effort(self) -> str:
        return str(self.attrs.get("reasoning_effort", "high"))

    @property
    def auto_status(self) -> bool:
        val = self.attrs.get("auto_status", False)
        return val.lower() == "true" if isinstance(val, str) else bool(val)

    @property
    def allow_partial(self) -> bool:
        val = self.attrs.get("allow_partial", False)
        return val.lower() == "true" if isinstance(val, str) else bool(val)


class Edge(BaseModel):
    model_config = ConfigDict(extra="allow")

    from_node: str
    to_node: str
    attrs: dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return str(self.attrs.get("label", ""))

    @property
    def condition(self) -> str:
        return str(self.attrs.get("condition", ""))

    @property
    def weight(self) -> int:
        return int(self.attrs.get("weight", 0))

    @property
    def fidelity(self) -> str:
        return str(self.attrs.get("fidelity", ""))

    @property
    def thread_id(self) -> str:
        return str(self.attrs.get("thread_id", ""))

    @property
    def loop_restart(self) -> bool:
        val = self.attrs.get("loop_restart", False)
        return val.lower() == "true" if isinstance(val, str) else bool(val)


class Graph(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    attrs: dict[str, Any] = Field(default_factory=dict)
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)

    @property
    def goal(self) -> str:
        return str(self.attrs.get("goal", ""))

    @property
    def label(self) -> str:
        return str(self.attrs.get("label", self.name))

    @property
    def model_stylesheet(self) -> str:
        return str(self.attrs.get("model_stylesheet", ""))

    @property
    def default_max_retries(self) -> int:
        if "default_max_retries" in self.attrs:
            return int(self.attrs["default_max_retries"])
        if "default_max_retry" in self.attrs:
            return int(self.attrs["default_max_retry"])
        return 0

    @property
    def retry_target(self) -> str:
        return str(self.attrs.get("retry_target", ""))

    @property
    def fallback_retry_target(self) -> str:
        return str(self.attrs.get("fallback_retry_target", ""))

    @property
    def default_fidelity(self) -> str:
        return str(self.attrs.get("default_fidelity", ""))

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.to_node == node_id]


class Outcome(BaseModel):
    status: StageStatus = StageStatus.SUCCESS
    preferred_label: str = ""
    suggested_next_ids: list[str] = Field(default_factory=list)
    context_updates: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""
    completed_nodes: list[str] = Field(default_factory=list)

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.status.value),
            "preferred_label": self.preferred_label,
            "suggested_next_ids": self.suggested_next_ids,
            "context_updates": self.context_updates,
            "notes": self.notes,
            "failure_reason": self.failure_reason,
        }


class Context:
    """Thread-safe key-value context store shared across stages in a pipeline run."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._values: dict[str, Any] = copy.deepcopy(initial) if initial else {}
        self._logs: list[str] = []
        self._lock = threading.RLock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def get_string(self, key: str, default: str = "") -> str:
        val = self.get(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return "true" if val else "false"
        return str(val)

    def append_log(self, entry: str) -> None:
        with self._lock:
            self._logs.append(entry)

    @property
    def logs(self) -> list[str]:
        with self._lock:
            return list(self._logs)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._values)

    def clone(self) -> Context:
        with self._lock:
            cloned = Context(copy.deepcopy(self._values))
            cloned._logs = list(self._logs)
            return cloned

    def apply_updates(self, updates: dict[str, Any]) -> None:
        with self._lock:
            for k, v in updates.items():
                self._values[k] = v


class Checkpoint(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    current_node: str
    completed_nodes: list[str] = Field(default_factory=list)
    node_retries: dict[str, int] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Checkpoint:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


class Diagnostic(BaseModel):
    rule: str
    severity: Severity
    message: str
    node_id: str | None = None
    edge: tuple[str, str] | None = None
    fix: str | None = None


class Option(BaseModel):
    key: str
    label: str
    to_node: str = ""


class Question(BaseModel):
    id: str = ""
    text: str
    type: QuestionType = QuestionType.MULTIPLE_CHOICE
    options: list[Option] = Field(default_factory=list)
    default: str | None = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    value: str
    selected_option: Option | None = None
    text: str = ""


class PipelineEvent(BaseModel):
    type: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_id: str = ""
    node_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
