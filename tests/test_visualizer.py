"""Tests for Attractor Pipeline Visualizer & Diagram Exporter."""

from __future__ import annotations

import json
from attractor.mcp_server import AttractorMcpServer
from attractor.visualizer import dot_to_ascii, dot_to_mermaid

SAMPLE_DOT = """
digraph Pipeline {
    graph [label="Sample Factory Pipeline"];
    start [shape=Mdiamond, label="Start"];
    build [shape=box, label="Compile Code"];
    gate [shape=diamond, label="Tests Passing?"];
    exit [shape=Msquare, label="Complete"];

    start -> build;
    build -> gate;
    gate -> exit [label="yes", condition="outcome = pass"];
    gate -> build [label="no", condition="outcome = fail"];
}
"""


def test_dot_to_mermaid_export() -> None:
    mermaid_code = dot_to_mermaid(SAMPLE_DOT)
    assert mermaid_code.startswith("graph LR")
    assert 'start(["Start"])' in mermaid_code
    assert 'build["Compile Code"]' in mermaid_code
    assert 'gate{"Tests Passing?"}' in mermaid_code
    assert 'exit[["Complete"]]' in mermaid_code
    assert "start --> build" in mermaid_code
    assert "outcome = pass" in mermaid_code


def test_dot_to_ascii_export() -> None:
    ascii_art = dot_to_ascii(SAMPLE_DOT)
    assert "Sample Factory Pipeline" in ascii_art
    assert "STAGES & TOPOLOGY:" in ascii_art
    assert "[start] <Mdiamond>: Start" in ascii_art
    assert "gate ---> exit (yes) [cond: outcome = pass]" in ascii_art


def test_mcp_export_diagram_tool() -> None:
    server = AttractorMcpServer()

    # 1. Mermaid format
    req_mermaid = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "attractor_export_diagram",
            "arguments": {"dot_source": SAMPLE_DOT, "format": "mermaid"},
        },
    }
    resp1 = server.handle_request(req_mermaid)
    assert resp1 is not None
    assert "graph LR" in resp1["result"]["content"][0]["text"]

    # 2. ASCII format
    req_ascii = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "attractor_export_diagram",
            "arguments": {"dot_source": SAMPLE_DOT, "format": "ascii"},
        },
    }
    resp2 = server.handle_request(req_ascii)
    assert resp2 is not None
    assert "TRANSITIONS & GATES:" in resp2["result"]["content"][0]["text"]
