"""Attractor DOT subset parser (Section 2 of attractor-spec.md)."""

from __future__ import annotations

import copy
import re
from typing import Any

from .models import Edge, Graph, Node


class DotParseError( ValueError ):
    """Raised when a DOT source violates the Attractor DOT DSL grammar."""


def strip_comments(source: str) -> str:
    """Strip // line comments and /* block */ comments outside double-quoted strings."""
    result: list[str] = []
    i = 0
    n = len(source)
    in_string = False
    while i < n:
        ch = source[i]
        if in_string:
            result.append(ch)
            if ch == "\\" and i + 1 < n:
                i += 1
                result.append(source[i])
            elif ch == '"':
                in_string = False
            i += 1
        else:
            if ch == '"':
                in_string = True
                result.append(ch)
                i += 1
            elif source[i : i + 2] == "//":
                i += 2
                while i < n and source[i] != "\n":
                    i += 1
            elif source[i : i + 2] == "/*":
                i += 2
                while i < n and source[i : i + 2] != "*/":
                    i += 1
                i += 2
            else:
                result.append(ch)
                i += 1
    return "".join(result)


_TOKEN_SPEC = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<ARROW>->)
  | (?P<UNDirected>--)
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<LBRACKET>\[)
  | (?P<RBRACKET>\])
  | (?P<EQUALS>=)
  | (?P<COMMA>,)
  | (?P<SEMI>;)
  | (?P<STRING>"(?:\\.|[^"\\])*")
  | (?P<BARE>[A-Za-z0-9_.:-]+)
""",
    re.VERBOSE,
)


class Token:
    __slots__ = ("kind", "value", "pos")

    def __init__(self, kind: str, value: str, pos: int) -> None:
        self.kind = kind
        self.value = value
        self.pos = pos

    def __repr__(self) -> str:
        return f"Token({self.kind!r}, {self.value!r})"


def tokenize(source: str) -> list[Token]:
    cleaned = strip_comments(source)
    tokens: list[Token] = []
    pos = 0
    n = len(cleaned)
    while pos < n:
        m = _TOKEN_SPEC.match(cleaned, pos)
        if not m:
            snippet = cleaned[pos : pos + 20]
            raise DotParseError(f"Unexpected character at position {pos}: {snippet!r}")
        kind = m.lastgroup or ""
        val = m.group()
        if kind == "UNDirected":
            raise DotParseError("Undirected edges ('--') are not supported in Attractor DOT.")
        if kind != "WS":
            tokens.append(Token(kind, val, pos))
        pos = m.end()
    return tokens


def parse_typed_value(token: Token) -> Any:
    if token.kind == "STRING":
        inner = token.value[1:-1]
        inner = (
            inner.replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\\\", "\\")
        )
        return inner
    raw = token.value
    if raw == "true":
        return True
    if raw == "false":
        return False
    if re.fullmatch(r"-?[0-9]+", raw):
        return int(raw)
    if re.fullmatch(r"-?[0-9]*\.[0-9]+", raw):
        return float(raw)
    return raw


def _derive_class_from_label(label: str) -> str:
    slug = label.lower().replace(" ", "-")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    return slug.strip("-")


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.idx = 0

    def peek(self, offset: int = 0) -> Token | None:
        pos = self.idx + offset
        if 0 <= pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def consume(self, expected_kind: str | None = None, expected_val: str | None = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise DotParseError("Unexpected end of DOT input")
        if expected_kind and tok.kind != expected_kind:
            raise DotParseError(
                f"Expected token {expected_kind}, got {tok.kind} ({tok.value!r}) at pos {tok.pos}"
            )
        if expected_val and tok.value != expected_val:
            raise DotParseError(
                f"Expected {expected_val!r}, got {tok.value!r} at pos {tok.pos}"
            )
        self.idx += 1
        return tok

    def parse_graph(self) -> Graph:
        first = self.peek()
        if first is None:
            raise DotParseError("Empty DOT source")
        if first.value == "strict":
            raise DotParseError("The 'strict' modifier is rejected by the Attractor spec.")
        if first.value != "digraph":
            raise DotParseError(f"Expected 'digraph', got {first.value!r}")
        self.consume("BARE", "digraph")

        name_tok = self.peek()
        if name_tok and name_tok.kind in ("BARE", "STRING"):
            self.consume()
            graph_name = str(parse_typed_value(name_tok))
        else:
            graph_name = "Pipeline"

        self.consume("LBRACE")
        graph = Graph(name=graph_name)
        self._parse_statements(
            graph=graph,
            node_defaults={},
            edge_defaults={},
            inherited_classes=[],
            is_subgraph=False,
        )
        self.consume("RBRACE")
        if self.peek() is not None:
            raise DotParseError("Multiple graphs per file are not allowed.")
        return graph

    def _parse_attr_block(self) -> dict[str, Any]:
        self.consume("LBRACKET")
        attrs: dict[str, Any] = {}
        while True:
            tok = self.peek()
            if tok is None:
                raise DotParseError("Unclosed attribute block '['")
            if tok.kind == "RBRACKET":
                self.consume("RBRACKET")
                break
            key_tok = self.consume()
            key = str(parse_typed_value(key_tok))
            self.consume("EQUALS")
            val_tok = self.consume()
            attrs[key] = parse_typed_value(val_tok)
            sep = self.peek()
            if sep and sep.kind in ("COMMA", "SEMI"):
                self.consume()
        return attrs

    def _parse_statements(
        self,
        graph: Graph,
        node_defaults: dict[str, Any],
        edge_defaults: dict[str, Any],
        inherited_classes: list[str],
        is_subgraph: bool = False,
    ) -> None:
        local_node_defaults = copy.deepcopy(node_defaults)
        local_edge_defaults = copy.deepcopy(edge_defaults)
        local_classes = list(inherited_classes)
        subgraph_nodes_created: list[str] = []

        while True:
            tok = self.peek()
            if tok is None or tok.kind == "RBRACE":
                break
            if tok.kind == "SEMI":
                self.consume("SEMI")
                continue

            # 1. 'graph [ ... ]'
            if tok.kind == "BARE" and tok.value == "graph" and self.peek(1) and self.peek(1).kind == "LBRACKET":
                self.consume("BARE", "graph")
                attrs = self._parse_attr_block()
                if not is_subgraph:
                    graph.attrs.update(attrs)
                else:
                    if "label" in attrs:
                        derived = _derive_class_from_label(str(attrs["label"]))
                        if derived and derived not in local_classes:
                            local_classes.append(derived)
                continue

            # 2. 'node [ ... ]'
            if tok.kind == "BARE" and tok.value == "node" and self.peek(1) and self.peek(1).kind == "LBRACKET":
                self.consume("BARE", "node")
                attrs = self._parse_attr_block()
                local_node_defaults.update(attrs)
                continue

            # 3. 'edge [ ... ]'
            if tok.kind == "BARE" and tok.value == "edge" and self.peek(1) and self.peek(1).kind == "LBRACKET":
                self.consume("BARE", "edge")
                attrs = self._parse_attr_block()
                local_edge_defaults.update(attrs)
                continue

            # 4. 'subgraph [id] { ... }'
            if tok.kind == "BARE" and tok.value == "subgraph":
                self.consume("BARE", "subgraph")
                next_tok = self.peek()
                if next_tok and next_tok.kind != "LBRACE":
                    self.consume()
                self.consume("LBRACE")
                self._parse_statements(
                    graph=graph,
                    node_defaults=local_node_defaults,
                    edge_defaults=local_edge_defaults,
                    inherited_classes=local_classes,
                    is_subgraph=True,
                )
                self.consume("RBRACE")
                continue

            # 5. Top-level or subgraph 'key = value'
            if self.peek(1) and self.peek(1).kind == "EQUALS":
                key_tok = self.consume()
                self.consume("EQUALS")
                val_tok = self.consume()
                key = str(parse_typed_value(key_tok))
                val = parse_typed_value(val_tok)
                if is_subgraph and key == "label":
                    derived = _derive_class_from_label(str(val))
                    if derived and derived not in local_classes:
                        local_classes.append(derived)
                        for nid in subgraph_nodes_created:
                            if derived not in graph.nodes[nid].classes:
                                graph.nodes[nid].classes.append(derived)
                else:
                    graph.attrs[key] = val
                continue

            # 6. Edge statement ('A -> B (-> C)* [...]') or Node statement ('A [...]')
            first_id_tok = self.consume()
            first_id = str(parse_typed_value(first_id_tok))

            if self.peek() and self.peek().kind == "ARROW":
                chain_ids = [first_id]
                while self.peek() and self.peek().kind == "ARROW":
                    self.consume("ARROW")
                    target_tok = self.consume()
                    chain_ids.append(str(parse_typed_value(target_tok)))

                explicit_edge_attrs: dict[str, Any] = {}
                if self.peek() and self.peek().kind == "LBRACKET":
                    explicit_edge_attrs = self._parse_attr_block()

                merged_edge_attrs = {**local_edge_defaults, **explicit_edge_attrs}
                for nid in chain_ids:
                    if nid not in graph.nodes:
                        self._ensure_node(
                            graph,
                            nid,
                            local_node_defaults,
                            {},
                            local_classes,
                        )
                        subgraph_nodes_created.append(nid)
                for a, b in zip(chain_ids, chain_ids[1:]):
                    graph.edges.append(
                        Edge(from_node=a, to_node=b, attrs=copy.deepcopy(merged_edge_attrs))
                    )
            else:
                explicit_node_attrs: dict[str, Any] = {}
                if self.peek() and self.peek().kind == "LBRACKET":
                    explicit_node_attrs = self._parse_attr_block()
                self._ensure_node(
                    graph,
                    first_id,
                    local_node_defaults,
                    explicit_node_attrs,
                    local_classes,
                )
                subgraph_nodes_created.append(first_id)

    def _ensure_node(
        self,
        graph: Graph,
        node_id: str,
        defaults: dict[str, Any],
        explicit_attrs: dict[str, Any],
        inherited_classes: list[str],
    ) -> Node:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node_id):
            raise DotParseError(
                f"Invalid node identifier {node_id!r}. Must match [A-Za-z_][A-Za-z0-9_]*"
            )
        if node_id in graph.nodes:
            node = graph.nodes[node_id]
            for k, v in defaults.items():
                if k not in node.attrs:
                    node.attrs[k] = v
            node.attrs.update(explicit_attrs)
            node.explicit_attrs.update(explicit_attrs.keys())
        else:
            merged = {**defaults, **explicit_attrs}
            node = Node(
                id=node_id,
                attrs=merged,
                explicit_attrs=set(explicit_attrs.keys()),
            )
            graph.nodes[node_id] = node

        for c in inherited_classes:
            if c and c not in node.classes:
                node.classes.append(c)

        if "class" in node.attrs:
            raw_cls = str(node.attrs["class"])
            for part in raw_cls.split(","):
                c = part.strip()
                if c and c not in node.classes:
                    node.classes.append(c)
        return node


def parse_dot(source: str) -> Graph:
    """Parse an Attractor DOT string into a Graph model."""
    tokens = tokenize(source)
    parser = _Parser(tokens)
    return parser.parse_graph()
