"""Tests for Attractor Pipeline Complexity Analyzer & Semantic Diff Engine."""

from __future__ import annotations

import json
from fastapi.testclient import TestClient

from attractor.analyzer import analyze_pipeline, diff_pipelines
from attractor.api import app
from attractor.mcp_server import AttractorMcpServer

LINEAR_DOT = """
digraph Linear {
    start [shape=Mdiamond, label="Start"];
    plan [shape=box, label="Plan", prompt="Plan changes"];
    build [shape=box, label="Build", prompt="Build feature"];
    exit [shape=Msquare, label="Done"];
    start -> plan -> build -> exit;
}
"""

UNBOUNDED_CYCLE_DOT = """
digraph Risky {
    start [shape=Mdiamond, label="Start"];
    build [shape=box, label="Build", prompt="Write code"];
    gate [shape=diamond, label="Tests Pass?"];
    exit [shape=Msquare, label="Done"];
    start -> build -> gate;
    gate -> exit [label="yes", condition="outcome = success"];
    gate -> build [label="retry", condition="outcome = fail"];
}
"""

BOUNDED_CYCLE_DOT = """
digraph Safe {
    start [shape=Mdiamond, label="Start"];
    build [shape=box, label="Build", prompt="Write code", max_retries=3];
    gate [shape=diamond, label="Tests Pass?"];
    review [shape=hexagon, label="Human Review"];
    exit [shape=Msquare, label="Done"];
    start -> build -> gate;
    gate -> review [label="yes", condition="outcome = success"];
    review -> exit;
    gate -> build [label="retry", condition="outcome = fail"];
}
"""


def test_analyze_simple_linear_pipeline() -> None:
    report = analyze_pipeline(LINEAR_DOT)
    assert report["graph_name"] == "Linear"
    assert report["node_count"] == 4
    assert report["edge_count"] == 3
    assert report["cyclomatic_complexity"] == 1
    assert report["critical_path_length"] == 4
    assert report["llm_node_count"] == 2
    assert report["cycles"] == []
    assert report["risk_level"] == "low"


def test_analyze_unbounded_vs_bounded_cycle() -> None:
    risky = analyze_pipeline(UNBOUNDED_CYCLE_DOT)
    assert len(risky["cycles"]) == 1
    assert len(risky["unbounded_cycle_warnings"]) == 1
    assert risky["risk_level"] == "high"

    safe = analyze_pipeline(BOUNDED_CYCLE_DOT)
    assert len(safe["cycles"]) == 1
    assert safe["unbounded_cycle_warnings"] == []
    assert safe["human_gates"] == ["review"]
    assert safe["risk_level"] == "medium"


def test_diff_pipelines_semantic_changes() -> None:
    same = diff_pipelines(LINEAR_DOT, LINEAR_DOT)
    assert same["identical"] is True
    assert same["has_breaking_changes"] is False

    diff = diff_pipelines(UNBOUNDED_CYCLE_DOT, BOUNDED_CYCLE_DOT)
    assert diff["identical"] is False
    assert diff["has_breaking_changes"] is True
    assert diff["added_nodes"] == ["review"]
    assert diff["removed_nodes"] == []
    assert any(m["node_id"] == "build" and "max_retries" in m["changes"] for m in diff["modified_nodes"])
    assert len(diff["added_edges"]) == 2
    assert len(diff["removed_edges"]) == 1


def test_mcp_analyze_and_diff_tools() -> None:
    server = AttractorMcpServer()

    resp_analyze = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "attractor_analyze_pipeline", "arguments": {"dot_source": BOUNDED_CYCLE_DOT}},
        }
    )
    assert resp_analyze is not None
    data = json.loads(resp_analyze["result"]["content"][0]["text"])
    assert data["human_gates"] == ["review"]

    resp_diff = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "attractor_diff_pipelines",
                "arguments": {"dot_before": UNBOUNDED_CYCLE_DOT, "dot_after": BOUNDED_CYCLE_DOT},
            },
        }
    )
    assert resp_diff is not None
    diff_data = json.loads(resp_diff["result"]["content"][0]["text"])
    assert diff_data["added_nodes"] == ["review"]


def test_fastapi_analyze_and_diff_endpoints() -> None:
    client = TestClient(app)

    r1 = client.post("/api/pipelines/analyze", json={"dot_source": LINEAR_DOT})
    assert r1.status_code == 200
    assert r1.json()["risk_level"] == "low"

    r2 = client.post(
        "/api/pipelines/diff",
        json={"dot_before": UNBOUNDED_CYCLE_DOT, "dot_after": BOUNDED_CYCLE_DOT},
    )
    assert r2.status_code == 200
    assert r2.json()["added_nodes"] == ["review"]
