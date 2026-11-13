"""Pipeline visualizer converting Attractor DOT graphs to Mermaid and ASCII diagrams."""

from __future__ import annotations

import re
from typing import Literal

from .models import Graph, Node
from .parser import parse_dot


def _escape_mermaid(text: str) -> str:
    """Escapes special characters in Mermaid node labels."""
    return text.replace('"', "'").replace("\n", " ").strip()


def dot_to_mermaid(dot_or_graph: str | Graph, direction: Literal["LR", "TD"] = "LR") -> str:
    """Converts an Attractor DOT pipeline into GitHub-compatible Mermaid flowchart markdown."""
    if isinstance(dot_or_graph, str):
        graph = parse_dot(dot_or_graph)
    else:
        graph = dot_or_graph

    lines = [f"graph {direction}"]

    # Define custom node styles
    lines.append("    %% Node Definitions")
    for node_id, node in graph.nodes.items():
        label = _escape_mermaid(node.label or node_id)
        shape = node.shape or "box"

        if shape == "Mdiamond":
            node_repr = f'    {node_id}(["{label}"])'
        elif shape == "Msquare":
            node_repr = f'    {node_id}[["{label}"]]'
        elif shape == "diamond":
            node_repr = f'    {node_id}{{"{label}"}}'
        elif shape == "hexagon":
            node_repr = f'    {node_id}{{{{"{label}"}}}}'
        elif shape == "component":
            node_repr = f'    {node_id}[/"{label}"/]'
        else:
            node_repr = f'    {node_id}["{label}"]'

        lines.append(node_repr)

    lines.append("\n    %% Edge Transitions")
    for edge in graph.edges:
        source = edge.from_node
        target = edge.to_node
        lbl = edge.label or ""
        cond = edge.condition or ""

        edge_label = ""
        if lbl and cond:
            edge_label = f"|{_escape_mermaid(lbl)} ({_escape_mermaid(cond)})|"
        elif lbl:
            edge_label = f"|{_escape_mermaid(lbl)}|"
        elif cond:
            edge_label = f"|{_escape_mermaid(cond)}|"

        lines.append(f"    {source} -->{edge_label} {target}")

    # Color classes
    lines.append("\n    %% Styles")
    lines.append("    classDef startClass fill:#10b981,stroke:#059669,stroke-width:2px,color:#ffffff;")
    lines.append("    classDef exitClass fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#ffffff;")
    lines.append("    classDef gateClass fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#ffffff;")
    lines.append("    classDef taskClass fill:#0ea5e9,stroke:#0284c7,stroke-width:1px,color:#ffffff;")

    for node_id, node in graph.nodes.items():
        shape = node.shape or "box"
        if shape == "Mdiamond":
            lines.append(f"    class {node_id} startClass;")
        elif shape == "Msquare":
            lines.append(f"    class {node_id} exitClass;")
        elif shape == "diamond":
            lines.append(f"    class {node_id} gateClass;")
        else:
            lines.append(f"    class {node_id} taskClass;")

    return "\n".join(lines)


def dot_to_ascii(dot_or_graph: str | Graph) -> str:
    """Generates a structured ASCII summary flow for terminal display."""
    if isinstance(dot_or_graph, str):
        graph = parse_dot(dot_or_graph)
    else:
        graph = dot_or_graph

    title = graph.attrs.get("label", graph.name) or "Attractor Software Factory Pipeline"
    header = f"=== [ {title} ] ==="
    sep = "=" * len(header)
    lines = [sep, header, sep, ""]

    # List stages in topology or node order
    lines.append("STAGES & TOPOLOGY:")
    for node_id, node in graph.nodes.items():
        shape_type = node.shape or "box"
        lbl = node.label or node_id
        prompt_preview = f" (prompt: {node.prompt[:30]}...)" if node.prompt else ""
        lines.append(f"  * [{node_id}] <{shape_type}>: {lbl}{prompt_preview}")

    lines.append("\nTRANSITIONS & GATES:")
    for edge in graph.edges:
        cond_str = f" [cond: {edge.condition}]" if edge.condition else ""
        lbl_str = f" ({edge.label})" if edge.label else ""
        lines.append(f"  {edge.from_node} ---> {edge.to_node}{lbl_str}{cond_str}")

    lines.append(sep)
    return "\n".join(lines)
