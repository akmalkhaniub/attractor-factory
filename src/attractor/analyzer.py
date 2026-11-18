"""Static Complexity, Cycle Risk Analyzer & Semantic Diff Engine for Attractor DOT Pipelines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import SHAPE_TO_HANDLER_TYPE, Graph
from .parser import parse_dot


def _resolve_graph(dot_or_graph: str | Graph) -> Graph:
    if isinstance(dot_or_graph, str):
        return parse_dot(dot_or_graph)
    return dot_or_graph


def _detect_cycles(graph: Graph) -> list[list[str]]:
    """Detect simple directed cycles in the pipeline graph via DFS."""
    adj: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges:
        adj[e.from_node].append(e.to_node)

    visited: set[str] = set()
    in_stack: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []
    seen_canonical: set[tuple[str, ...]] = set()

    def dfs(u: str) -> None:
        visited.add(u)
        in_stack.add(u)
        stack.append(u)
        for v in adj.get(u, []):
            if v not in visited:
                dfs(v)
            elif v in in_stack:
                idx = stack.index(v)
                cycle = stack[idx:] + [v]
                # Canonicalize cycle representation by minimum node
                core = cycle[:-1]
                min_i = core.index(min(core))
                canon = tuple(core[min_i:] + core[:min_i])
                if canon not in seen_canonical:
                    seen_canonical.add(canon)
                    cycles.append(cycle)
        stack.pop()
        in_stack.remove(u)

    for nid in sorted(graph.nodes.keys()):
        if nid not in visited:
            dfs(nid)

    return cycles


def _longest_simple_path(graph: Graph) -> int:
    """Compute longest simple path length (number of nodes) from start nodes."""
    adj: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges:
        adj[e.from_node].append(e.to_node)

    start_nodes = [n.id for n in graph.nodes.values() if n.shape == "Mdiamond"] or list(graph.nodes.keys())[:1]
    best = 0

    def dfs(u: str, seen: set[str]) -> int:
        max_depth = 1
        for v in adj.get(u, []):
            if v not in seen and v in graph.nodes:
                seen.add(v)
                d = 1 + dfs(v, seen)
                if d > max_depth:
                    max_depth = d
                seen.remove(v)
        return max_depth

    for s in start_nodes:
        depth = dfs(s, {s})
        if depth > best:
            best = depth
    return best


def analyze_pipeline(dot_or_graph: str | Graph) -> dict[str, Any]:
    """Perform static complexity, cycle risk, and estimated cost analysis on a DOT pipeline."""
    graph = _resolve_graph(dot_or_graph)
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    cyclomatic = max(1, edge_count - node_count + 2)

    llm_nodes: list[str] = []
    human_gates: list[str] = []
    conditional_gates: list[str] = []
    parallel_nodes: list[str] = []

    for nid, node in graph.nodes.items():
        handler = node.node_type or SHAPE_TO_HANDLER_TYPE.get(node.shape or "box", "codergen")
        if handler == "codergen":
            llm_nodes.append(nid)
        elif handler == "wait.human":
            human_gates.append(nid)
        elif handler == "conditional":
            conditional_gates.append(nid)
        elif handler == "parallel":
            parallel_nodes.append(nid)

    cycles = _detect_cycles(graph)
    unbounded_warnings: list[str] = []
    for cyc in cycles:
        core_nodes = [graph.nodes[nid] for nid in cyc[:-1] if nid in graph.nodes]
        has_retry_cap = any(n.max_retries is not None and n.max_retries > 0 for n in core_nodes)
        if not has_retry_cap:
            unbounded_warnings.append(
                f"Cycle {' -> '.join(cyc)} has no explicit max_retries cap; potential infinite retry loop."
            )

    critical_path = _longest_simple_path(graph)
    est_tokens = len(llm_nodes) * 4000 + len(cycles) * 2500

    if unbounded_warnings or cyclomatic >= 8:
        risk_level = "high"
    elif cycles or cyclomatic >= 4:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "graph_name": graph.name,
        "node_count": node_count,
        "edge_count": edge_count,
        "cyclomatic_complexity": cyclomatic,
        "critical_path_length": critical_path,
        "llm_nodes": sorted(llm_nodes),
        "llm_node_count": len(llm_nodes),
        "human_gates": sorted(human_gates),
        "conditional_gates": sorted(conditional_gates),
        "parallel_nodes": sorted(parallel_nodes),
        "cycles": cycles,
        "unbounded_cycle_warnings": unbounded_warnings,
        "estimated_token_budget": est_tokens,
        "risk_level": risk_level,
    }


def diff_pipelines(dot_a: str | Graph, dot_b: str | Graph) -> dict[str, Any]:
    """Compute a structured semantic AST diff between two Attractor DOT pipelines."""
    g_a = _resolve_graph(dot_a)
    g_b = _resolve_graph(dot_b)

    nodes_a = set(g_a.nodes.keys())
    nodes_b = set(g_b.nodes.keys())

    added_nodes = sorted(nodes_b - nodes_a)
    removed_nodes = sorted(nodes_a - nodes_b)

    modified_nodes: list[dict[str, Any]] = []
    for nid in sorted(nodes_a & nodes_b):
        na = g_a.nodes[nid]
        nb = g_b.nodes[nid]
        changes: dict[str, dict[str, Any]] = {}
        all_keys = sorted(set(na.attrs.keys()) | set(nb.attrs.keys()))
        for k in all_keys:
            va = na.attrs.get(k)
            vb = nb.attrs.get(k)
            if va != vb:
                changes[k] = {"before": va, "after": vb}
        if changes:
            modified_nodes.append({"node_id": nid, "changes": changes})

    def _edge_sig(e: Any) -> tuple[str, str, str, str]:
        return (e.from_node, e.to_node, e.label, e.condition)

    sigs_a = {_edge_sig(e) for e in g_a.edges}
    sigs_b = {_edge_sig(e) for e in g_b.edges}

    added_edges = [
        {"from_node": s[0], "to_node": s[1], "label": s[2], "condition": s[3]}
        for s in sorted(sigs_b - sigs_a)
    ]
    removed_edges = [
        {"from_node": s[0], "to_node": s[1], "label": s[2], "condition": s[3]}
        for s in sorted(sigs_a - sigs_b)
    ]

    has_breaking = bool(removed_nodes or removed_edges)
    is_identical = not (added_nodes or removed_nodes or modified_nodes or added_edges or removed_edges)

    summary = (
        "Pipelines are semantically identical."
        if is_identical
        else (
            f"+{len(added_nodes)}/-{len(removed_nodes)}/~{len(modified_nodes)} nodes, "
            f"+{len(added_edges)}/-{len(removed_edges)} edges"
        )
    )

    return {
        "identical": is_identical,
        "has_breaking_changes": has_breaking,
        "summary": summary,
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "modified_nodes": modified_nodes,
        "added_edges": added_edges,
        "removed_edges": removed_edges,
    }
