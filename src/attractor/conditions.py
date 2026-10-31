"""Condition Expression Language (Section 10 of attractor-spec.md)."""

from __future__ import annotations

import re

from .models import Context, Outcome


class ConditionSyntaxError(ValueError):
    """Raised when an edge condition expression is syntactically invalid."""


_VALID_CLAUSE_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_.]*)\s*(?:(?P<op>!=|=)\s*(?P<val>.+?))?\s*$"
)


def parse_literal(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return (
            value[1:-1]
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\\\", "\\")
        )
    return value


def validate_condition_syntax(condition: str) -> None:
    if not condition or not condition.strip():
        return
    if "||" in condition:
        raise ConditionSyntaxError(f"Unsupported operator '||' in condition: {condition!r}")
    clauses = condition.split("&&")
    for raw_clause in clauses:
        clause = raw_clause.strip()
        if not clause:
            raise ConditionSyntaxError(f"Empty clause in condition: {condition!r}")
        for invalid_op in ("==", ">=", "<=", ">", "<"):
            if invalid_op in clause and "!=" not in clause:
                raise ConditionSyntaxError(
                    f"Invalid operator {invalid_op!r} in condition {condition!r}"
                )
        m = _VALID_CLAUSE_RE.match(clause)
        if not m:
            raise ConditionSyntaxError(f"Invalid condition clause: {clause!r}")
        op = m.group("op")
        val = m.group("val")
        if op and (val is None or not val.strip()):
            raise ConditionSyntaxError(f"Missing right-hand value in clause: {clause!r}")


def resolve_key(key: str, outcome: Outcome, context: Context) -> str:
    if key == "outcome":
        return str(outcome.status.value)
    if key == "preferred_label":
        return outcome.preferred_label or context.get_string("preferred_label", "")
    if key.startswith("context."):
        val = context.get(key)
        if val is not None:
            return "true" if val is True else ("false" if val is False else str(val))
        stripped = key[len("context.") :]
        val = context.get(stripped)
        if val is not None:
            return "true" if val is True else ("false" if val is False else str(val))
        return ""
    val = context.get(key)
    if val is not None:
        return "true" if val is True else ("false" if val is False else str(val))
    return ""


def evaluate_clause(clause: str, outcome: Outcome, context: Context) -> bool:
    if "!=" in clause:
        key, value = clause.split("!=", 1)
        return resolve_key(key.strip(), outcome, context) != parse_literal(value)
    if "=" in clause:
        key, value = clause.split("=", 1)
        return resolve_key(key.strip(), outcome, context) == parse_literal(value)
    resolved = resolve_key(clause.strip(), outcome, context)
    return bool(resolved and resolved.lower() not in ("false", "0", ""))


def evaluate_condition(condition: str, outcome: Outcome, context: Context) -> bool:
    if not condition or not condition.strip():
        return True
    clauses = condition.split("&&")
    for raw in clauses:
        clause = raw.strip()
        if not clause:
            continue
        if not evaluate_clause(clause, outcome, context):
            return False
    return True
