"""Validation and Linting (Section 7 of attractor-spec.md)."""

from __future__ import annotations

from collections import deque
from typing import Protocol

from .conditions import ConditionSyntaxError, validate_condition_syntax
from .models import (
    SHAPE_TO_HANDLER_TYPE,
    VALID_FIDELITY_MODES,
    Diagnostic,
    Graph,
    Node,
    Severity,
)
from .transforms import StylesheetSyntaxError, parse_stylesheet


class ValidationError(Exception):
    """Raised by validate_or_raise when error-severity diagnostics exist."""

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        messages = "; ".join(f"[{d.rule}] {d.message}" for d in diagnostics if d.severity == Severity.ERROR)
        super().__init__(f"Pipeline validation failed: {messages}")


class LintRule(Protocol):
    name: str

    def apply(self, graph: Graph) -> list[Diagnostic]: ...


def find_start_nodes(graph: Graph) -> list[Node]:
    by_shape = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
    if by_shape:
        return by_shape
    return [n for n in graph.nodes.values() if n.id in ("start", "Start")]


def find_terminal_nodes(graph: Graph) -> list[Node]:
    by_shape = [n for n in graph.nodes.values() if n.shape == "Msquare"]
    if by_shape:
        return by_shape
    return [n for n in graph.nodes.values() if n.id in ("exit", "end", "done")]


class StartNodeRule:
    name = "start_node"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        starts = find_start_nodes(graph)
        if len(starts) == 0:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    message="Pipeline must have exactly one start node (shape=Mdiamond or id='start').",
                    fix="Add 'start [shape=Mdiamond]' to the graph.",
                )
            ]
        if len(starts) > 1:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    message=f"Pipeline has multiple start nodes: {[n.id for n in starts]}.",
                    fix="Keep only one start node.",
                )
            ]
        return []


class TerminalNodeRule:
    name = "terminal_node"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        exits = find_terminal_nodes(graph)
        if len(exits) == 0:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    message="Pipeline must have exactly one terminal node (shape=Msquare or id='exit').",
                    fix="Add 'exit [shape=Msquare]' to the graph.",
                )
            ]
        if len(exits) > 1:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    message=f"Pipeline has multiple terminal nodes: {[n.id for n in exits]}.",
                    fix="Keep only one terminal node.",
                )
            ]
        return []


class EdgeTargetExistsRule:
    name = "edge_target_exists"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for edge in graph.edges:
            if edge.from_node not in graph.nodes:
                diags.append(
                    Diagnostic(
                        rule=self.name,
                        severity=Severity.ERROR,
                        message=f"Edge source '{edge.from_node}' does not exist.",
                        edge=(edge.from_node, edge.to_node),
                    )
                )
            if edge.to_node not in graph.nodes:
                diags.append(
                    Diagnostic(
                        rule=self.name,
                        severity=Severity.ERROR,
                        message=f"Edge target '{edge.to_node}' does not exist.",
                        edge=(edge.from_node, edge.to_node),
                    )
                )
        return diags


class StartNoIncomingRule:
    name = "start_no_incoming"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        starts = find_start_nodes(graph)
        if len(starts) != 1:
            return []
        start = starts[0]
        incoming = graph.incoming_edges(start.id)
        if incoming:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    node_id=start.id,
                    message=f"Start node '{start.id}' must not have incoming edges.",
                )
            ]
        return []


class ExitNoOutgoingRule:
    name = "exit_no_outgoing"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        exits = find_terminal_nodes(graph)
        if len(exits) != 1:
            return []
        exit_node = exits[0]
        outgoing = graph.outgoing_edges(exit_node.id)
        if outgoing:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    node_id=exit_node.id,
                    message=f"Terminal node '{exit_node.id}' must not have outgoing edges.",
                )
            ]
        return []


class ReachabilityRule:
    name = "reachability"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        starts = find_start_nodes(graph)
        if len(starts) != 1:
            return []
        start_id = starts[0].id
        visited: set[str] = {start_id}
        queue: deque[str] = deque([start_id])
        while queue:
            curr = queue.popleft()
            for edge in graph.outgoing_edges(curr):
                if edge.to_node in graph.nodes and edge.to_node not in visited:
                    visited.add(edge.to_node)
                    queue.append(edge.to_node)

        diags: list[Diagnostic] = []
        for nid in graph.nodes:
            if nid not in visited:
                diags.append(
                    Diagnostic(
                        rule=self.name,
                        severity=Severity.ERROR,
                        node_id=nid,
                        message=f"Node '{nid}' is unreachable from start node '{start_id}'.",
                        fix=f"Connect '{nid}' to the pipeline graph or remove it.",
                    )
                )
        return diags


class ConditionSyntaxRule:
    name = "condition_syntax"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for edge in graph.edges:
            if edge.condition:
                try:
                    validate_condition_syntax(edge.condition)
                except ConditionSyntaxError as exc:
                    diags.append(
                        Diagnostic(
                            rule=self.name,
                            severity=Severity.ERROR,
                            edge=(edge.from_node, edge.to_node),
                            message=str(exc),
                        )
                    )
        return diags


class StylesheetSyntaxRule:
    name = "stylesheet_syntax"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        if not graph.model_stylesheet:
            return []
        try:
            parse_stylesheet(graph.model_stylesheet)
        except StylesheetSyntaxError as exc:
            return [
                Diagnostic(
                    rule=self.name,
                    severity=Severity.ERROR,
                    message=str(exc),
                )
            ]
        return []


class TypeKnownRule:
    name = "type_known"

    def __init__(self, known_types: set[str] | None = None) -> None:
        self.known_types = known_types or set(SHAPE_TO_HANDLER_TYPE.values())

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for node in graph.nodes.values():
            if node.node_type and node.node_type not in self.known_types:
                diags.append(
                    Diagnostic(
                        rule=self.name,
                        severity=Severity.WARNING,
                        node_id=node.id,
                        message=f"Node '{node.id}' has unrecognized handler type '{node.node_type}'.",
                    )
                )
        return diags


class FidelityValidRule:
    name = "fidelity_valid"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        if graph.default_fidelity and graph.default_fidelity not in VALID_FIDELITY_MODES:
            diags.append(
                Diagnostic(
                    rule=self.name,
                    severity=Severity.WARNING,
                    message=f"Invalid graph default_fidelity '{graph.default_fidelity}'.",
                )
            )
        for node in graph.nodes.values():
            if node.fidelity and node.fidelity not in VALID_FIDELITY_MODES:
                diags.append(
                    Diagnostic(
                        rule=self.name,
                        severity=Severity.WARNING,
                        node_id=node.id,
                        message=f"Node '{node.id}' has invalid fidelity '{node.fidelity}'.",
                    )
                )
        for edge in graph.edges:
            if edge.fidelity and edge.fidelity not in VALID_FIDELITY_MODES:
                diags.append(
                    Diagnostic(
                        rule=self.name,
                        severity=Severity.WARNING,
                        edge=(edge.from_node, edge.to_node),
                        message=f"Edge {edge.from_node}->{edge.to_node} has invalid fidelity '{edge.fidelity}'.",
                    )
                )
        return diags


class RetryTargetExistsRule:
    name = "retry_target_exists"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for attr in ("retry_target", "fallback_retry_target"):
            target = getattr(graph, attr)
            if target and target not in graph.nodes:
                diags.append(
                    Diagnostic(
                        rule=self.name,
                        severity=Severity.WARNING,
                        message=f"Graph {attr} '{target}' does not reference an existing node.",
                    )
                )
        for node in graph.nodes.values():
            for attr in ("retry_target", "fallback_retry_target"):
                target = getattr(node, attr)
                if target and target not in graph.nodes:
                    diags.append(
                        Diagnostic(
                            rule=self.name,
                            severity=Severity.WARNING,
                            node_id=node.id,
                            message=f"Node '{node.id}' {attr} '{target}' does not exist.",
                        )
                    )
        return diags


class GoalGateHasRetryRule:
    name = "goal_gate_has_retry"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for node in graph.nodes.values():
            if node.goal_gate:
                has_retry = bool(
                    node.retry_target
                    or node.fallback_retry_target
                    or graph.retry_target
                    or graph.fallback_retry_target
                )
                if not has_retry:
                    diags.append(
                        Diagnostic(
                            rule=self.name,
                            severity=Severity.WARNING,
                            node_id=node.id,
                            message=f"Goal gate node '{node.id}' has no retry_target or fallback_retry_target.",
                        )
                    )
        return diags


class PromptOnLlmNodesRule:
    name = "prompt_on_llm_nodes"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        starts = {n.id for n in find_start_nodes(graph)}
        exits = {n.id for n in find_terminal_nodes(graph)}
        for node in graph.nodes.values():
            if node.id in starts or node.id in exits:
                continue
            resolved_type = node.node_type or SHAPE_TO_HANDLER_TYPE.get(node.shape, "codergen")
            if resolved_type == "codergen":
                has_prompt = bool(node.prompt.strip())
                has_explicit_label = "label" in node.explicit_attrs and bool(str(node.attrs.get("label", "")).strip())
                if not has_prompt and not has_explicit_label:
                    diags.append(
                        Diagnostic(
                            rule=self.name,
                            severity=Severity.WARNING,
                            node_id=node.id,
                            message=f"Codergen node '{node.id}' should define a 'prompt' or 'label' attribute.",
                        )
                    )
        return diags


BUILT_IN_RULES: list[LintRule] = [
    StartNodeRule(),
    TerminalNodeRule(),
    EdgeTargetExistsRule(),
    StartNoIncomingRule(),
    ExitNoOutgoingRule(),
    ReachabilityRule(),
    ConditionSyntaxRule(),
    StylesheetSyntaxRule(),
    TypeKnownRule(),
    FidelityValidRule(),
    RetryTargetExistsRule(),
    GoalGateHasRetryRule(),
    PromptOnLlmNodesRule(),
]


def validate(
    graph: Graph,
    extra_rules: list[LintRule] | None = None,
    known_types: set[str] | None = None,
) -> list[Diagnostic]:
    rules: list[LintRule] = []
    for r in BUILT_IN_RULES:
        if isinstance(r, TypeKnownRule) and known_types is not None:
            rules.append(TypeKnownRule(known_types))
        else:
            rules.append(r)
    if extra_rules:
        rules.extend(extra_rules)

    diagnostics: list[Diagnostic] = []
    for rule in rules:
        diagnostics.extend(rule.apply(graph))
    return diagnostics


def validate_or_raise(
    graph: Graph,
    extra_rules: list[LintRule] | None = None,
    known_types: set[str] | None = None,
) -> list[Diagnostic]:
    diagnostics = validate(graph, extra_rules=extra_rules, known_types=known_types)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    if errors:
        raise ValidationError(diagnostics)
    return diagnostics
