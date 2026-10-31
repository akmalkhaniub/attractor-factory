"""Human-in-the-Loop Interviewer Pattern (Section 6 of attractor-spec.md)."""

from __future__ import annotations

import re
import threading
import uuid
from collections import deque
from typing import Callable, Protocol

from .models import Answer, AnswerValue, Option, Question, QuestionType

_ACCEL_BRACKET_RE = re.compile(r"^\[([A-Za-z0-9])\]\s*(.+)$")
_ACCEL_PAREN_RE = re.compile(r"^([A-Za-z0-9])\)\s*(.+)$")
_ACCEL_DASH_RE = re.compile(r"^([A-Za-z0-9])\s*-\s*(.+)$")


def parse_accelerator_key(label: str) -> str:
    """Extract shortcut key from edge label per Section 4.6."""
    cleaned = label.strip()
    if not cleaned:
        return "1"
    for regex in (_ACCEL_BRACKET_RE, _ACCEL_PAREN_RE, _ACCEL_DASH_RE):
        m = regex.match(cleaned)
        if m:
            return m.group(1).upper()
    return cleaned[0].upper()


def normalize_label(label: str) -> str:
    """Normalize edge label: lowercase, trim whitespace, strip accelerator prefixes."""
    cleaned = label.strip()
    for regex in (_ACCEL_BRACKET_RE, _ACCEL_PAREN_RE, _ACCEL_DASH_RE):
        m = regex.match(cleaned)
        if m:
            cleaned = m.group(2).strip()
            break
    return cleaned.lower()


class Interviewer(Protocol):
    def ask(self, question: Question) -> Answer: ...

    def ask_multiple(self, questions: list[Question]) -> list[Answer]: ...

    def inform(self, message: str, stage: str = "") -> None: ...


class BaseInterviewer:
    def ask(self, question: Question) -> Answer:
        raise NotImplementedError

    def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        return [self.ask(q) for q in questions]

    def inform(self, message: str, stage: str = "") -> None:
        pass


class AutoApproveInterviewer(BaseInterviewer):
    """Always selects YES for yes/no questions and the first option for multiple choice."""

    def ask(self, question: Question) -> Answer:
        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return Answer(value=AnswerValue.YES)
        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            first = question.options[0]
            return Answer(value=first.key, selected_option=first, text=first.label)
        return Answer(value="auto-approved", text="auto-approved")


class QueueInterviewer(BaseInterviewer):
    """Reads answers from a pre-filled answer queue for deterministic testing."""

    def __init__(self, answers: list[Answer | str] | None = None) -> None:
        self._queue: deque[Answer] = deque()
        for item in answers or []:
            if isinstance(item, Answer):
                self._queue.append(item)
            else:
                self._queue.append(Answer(value=str(item), text=str(item)))

    def ask(self, question: Question) -> Answer:
        if self._queue:
            ans = self._queue.popleft()
            if question.options and ans.selected_option is None:
                for opt in question.options:
                    if (
                        opt.key.lower() == ans.value.lower()
                        or opt.label.lower() == ans.value.lower()
                        or normalize_label(opt.label) == normalize_label(ans.value)
                    ):
                        ans.selected_option = opt
                        break
            return ans
        return Answer(value=AnswerValue.SKIPPED)


class CallbackInterviewer(BaseInterviewer):
    """Delegates question answering to a provided callback function."""

    def __init__(self, callback: Callable[[Question], Answer | str]) -> None:
        self.callback = callback

    def ask(self, question: Question) -> Answer:
        res = self.callback(question)
        if isinstance(res, Answer):
            return res
        return Answer(value=str(res), text=str(res))


class RecordingInterviewer(BaseInterviewer):
    """Wraps another interviewer and records all question-answer pairs."""

    def __init__(self, inner: Interviewer) -> None:
        self.inner = inner
        self.recordings: list[tuple[Question, Answer]] = []

    def ask(self, question: Question) -> Answer:
        answer = self.inner.ask(question)
        self.recordings.append((question, answer))
        return answer


class WebInteractiveInterviewer(BaseInterviewer):
    """Thread-safe interviewer for HTTP/Web UI mode that blocks until answered via REST API."""

    def __init__(self, on_question: Callable[[Question], None] | None = None) -> None:
        self._pending: dict[str, tuple[Question, threading.Event]] = {}
        self._answers: dict[str, Answer] = {}
        self._lock = threading.Lock()
        self._on_question = on_question

    def get_pending_questions(self) -> list[Question]:
        with self._lock:
            return [q for q, _ in self._pending.values()]

    def submit_answer(self, question_id: str, value: str, text: str = "") -> bool:
        with self._lock:
            entry = self._pending.get(question_id)
            if not entry:
                return False
            question, evt = entry
            selected: Option | None = None
            for opt in question.options:
                if (
                    opt.key.lower() == value.lower()
                    or opt.label.lower() == value.lower()
                    or normalize_label(opt.label) == normalize_label(value)
                ):
                    selected = opt
                    break
            self._answers[question_id] = Answer(
                value=selected.key if selected else value,
                selected_option=selected,
                text=text or (selected.label if selected else value),
            )
            evt.set()
            return True

    def ask(self, question: Question) -> Answer:
        if not question.id:
            question.id = f"q-{uuid.uuid4().hex[:8]}"
        evt = threading.Event()
        with self._lock:
            self._pending[question.id] = (question, evt)
        if self._on_question:
            self._on_question(question)

        timeout = question.timeout_seconds
        signaled = evt.wait(timeout=timeout)
        with self._lock:
            self._pending.pop(question.id, None)
            if not signaled:
                return Answer(value=AnswerValue.TIMEOUT)
            return self._answers.pop(question.id, Answer(value=AnswerValue.SKIPPED))
