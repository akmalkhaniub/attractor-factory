"""Model Stylesheet & AST Transforms (Sections 8 & 9 of attractor-spec.md)."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Protocol

from .models import Graph, Node

RECOGNIZED_STYLE_PROPERTIES = {"llm_model", "llm_provider", "reasoning_effort"}


class StylesheetSyntaxError(ValueError):
    """Raised when a model_stylesheet contains invalid syntax."""


@dataclass
class StyleRule:
    selector: str
    specificity: int
    order: int
    declarations: dict[str, str]


_RULE_RE = re.compile(r"([^{]+)\{([^}]*)\}", re.DOTALL)


def parse_stylesheet(stylesheet: str) -> list[StyleRule]:
    if not stylesheet or not stylesheet.strip():
        return []

    stripped = stylesheet.strip()
    if "{" not in stripped or "}" not in stripped:
        raise StylesheetSyntaxError("Stylesheet must contain selector { property: value } blocks.")
    if stripped.count("{") != stripped.count("}"):
        raise StylesheetSyntaxError("Unbalanced braces in model_stylesheet.")

    rules: list[StyleRule] = []
    pos = 0
    for idx, match in enumerate(_RULE_RE.finditer(stripped)):
        between = stripped[pos : match.start()].strip()
        if between:
            raise StylesheetSyntaxError(f"Unexpected text outside rule: {between!r}")
        pos = match.end()

        selector = match.group(1).strip()
        body = match.group(2).strip()
        if not selector:
            raise StylesheetSyntaxError("Empty selector in model_stylesheet")

        if selector == "*":
            specificity = 0
        elif selector.startswith("#"):
            if not re.fullmatch(r"#[A-Za-z_][A-Za-z0-9_]*", selector):
                raise StylesheetSyntaxError(f"Invalid ID selector: {selector!r}")
            specificity = 3
        elif selector.startswith("."):
            if not re.fullmatch(r"\.[a-z0-9-]+", selector):
                raise StylesheetSyntaxError(f"Invalid class selector: {selector!r}")
            specificity = 2
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", selector):
            specificity = 1
        else:
            raise StylesheetSyntaxError(f"Invalid stylesheet selector: {selector!r}")

        declarations: dict[str, str] = {}
        for part in body.split(";"):
            decl = part.strip()
            if not decl:
                continue
            if ":" not in decl:
                raise StylesheetSyntaxError(f"Invalid declaration (missing ':'): {decl!r}")
            prop, val = decl.split(":", 1)
            prop = prop.strip()
            val = val.strip().strip('"').strip("'")
            if prop not in RECOGNIZED_STYLE_PROPERTIES:
                raise StylesheetSyntaxError(f"Unrecognized stylesheet property: {prop!r}")
            if not val:
                raise StylesheetSyntaxError(f"Empty value for property {prop!r}")
            declarations[prop] = val

        if not declarations:
            raise StylesheetSyntaxError(f"Rule {selector!r} has no declarations")

        rules.append(
            StyleRule(
                selector=selector,
                specificity=specificity,
                order=idx,
                declarations=declarations,
            )
        )

    trailing = stripped[pos:].strip()
    if trailing:
        raise StylesheetSyntaxError(f"Unexpected trailing text in stylesheet: {trailing!r}")
    return rules


def _rule_matches_node(rule: StyleRule, node: Node) -> bool:
    sel = rule.selector
    if sel == "*":
        return True
    if sel.startswith("#"):
        return node.id == sel[1:]
    if sel.startswith("."):
        return sel[1:] in node.classes
    return node.shape == sel


class Transform(Protocol):
    def apply(self, graph: Graph) -> Graph: ...


class VariableExpansionTransform:
    """Expands $goal in node prompt attributes to the graph-level goal value."""

    def apply(self, graph: Graph) -> Graph:
        g = copy.deepcopy(graph)
        goal_val = g.goal
        for node in g.nodes.values():
            if "prompt" in node.attrs and "$goal" in str(node.attrs["prompt"]):
                node.attrs["prompt"] = str(node.attrs["prompt"]).replace("$goal", goal_val)
        return g


class StylesheetApplicationTransform:
    """Applies model_stylesheet rules to nodes according to specificity."""

    def apply(self, graph: Graph) -> Graph:
        g = copy.deepcopy(graph)
        raw_sheet = g.model_stylesheet
        if not raw_sheet or not raw_sheet.strip():
            return g
        try:
            rules = parse_stylesheet(raw_sheet)
        except StylesheetSyntaxError:
            # Leave syntax error to be caught and reported by the validator
            return g

        # Sort by (specificity ASC, order ASC) so later/higher-specificity rules overwrite earlier
        sorted_rules = sorted(rules, key=lambda r: (r.specificity, r.order))
        for node in g.nodes.values():
            resolved: dict[str, str] = {}
            for rule in sorted_rules:
                if _rule_matches_node(rule, node):
                    resolved.update(rule.declarations)
            for prop, val in resolved.items():
                if prop not in node.explicit_attrs:
                    node.attrs[prop] = val
        return g
