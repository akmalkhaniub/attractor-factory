"""Tests for the Attractor Model Context Protocol (MCP) Server."""

from __future__ import annotations

import json
from pathlib import Path

from attractor import AttractorMcpServer


def test_mcp_initialize_and_tools_list() -> None:
    server = AttractorMcpServer()

    # initialize
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    }
    init_resp = server.handle_request(init_req)
    assert init_resp["id"] == 1
    assert init_resp["result"]["serverInfo"]["name"] == "attractor-factory-mcp"
    assert "tools" in init_resp["result"]["capabilities"]

    # tools/list
    list_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    list_resp = server.handle_request(list_req)
    tools = list_resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "attractor_lint_pipeline" in tool_names
    assert "attractor_run_pipeline" in tool_names
    assert "attractor_list_presets" in tool_names
    assert "attractor_parse_graph" in tool_names


def test_mcp_tool_call_lint_and_presets() -> None:
    server = AttractorMcpServer()

    # list presets
    preset_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "attractor_list_presets",
            "arguments": {},
        },
    }
    preset_resp = server.handle_request(preset_req)
    assert not preset_resp["result"].get("isError")
    presets = json.loads(preset_resp["result"]["content"][0]["text"])
    assert len(presets) >= 4

    # lint valid pipeline
    dot = presets[0]["dot_source"]
    lint_req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "attractor_lint_pipeline",
            "arguments": {"dot_source": dot},
        },
    }
    lint_resp = server.handle_request(lint_req)
    lint_res = json.loads(lint_resp["result"]["content"][0]["text"])
    assert lint_res["valid"] is True


def test_mcp_tool_call_run_pipeline(tmp_path: Path) -> None:
    server = AttractorMcpServer()
    dot = """
    digraph MCPTest {
        graph [goal="Verify MCP pipeline run"]
        start [shape=Mdiamond]
        step1 [shape=box, prompt="Step 1"]
        exit  [shape=Msquare]
        start -> step1 -> exit
    }
    """
    run_req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "attractor_run_pipeline",
            "arguments": {
                "dot_source": dot,
                "workspace": str(tmp_path),
                "auto_approve": True,
            },
        },
    }
    run_resp = server.handle_request(run_req)
    assert not run_resp["result"].get("isError")
    run_res = json.loads(run_resp["result"]["content"][0]["text"])
    assert run_res["status"] == "success"
    assert "step1" in run_res["completed_nodes"]
