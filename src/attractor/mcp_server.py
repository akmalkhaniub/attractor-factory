"""Model Context Protocol (MCP) Server for Attractor Software Factory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .engine import PipelineRunner, run_pipeline
from .interviewer import AutoApproveInterviewer
from .models import Outcome, Severity, StageStatus
from .parser import DotParseError, parse_dot
from .presets import PRESET_PIPELINES
from .validator import validate
from .visualizer import dot_to_ascii, dot_to_mermaid

SERVER_NAME = "attractor-factory-mcp"
SERVER_VERSION = "0.1.0"

TOOL_DEFINITIONS = [
    {
        "name": "attractor_lint_pipeline",
        "description": "Lint and validate a Graphviz DOT software factory pipeline against the 13 Attractor specification rules.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dot_source": {
                    "type": "string",
                    "description": "The complete Graphviz DOT pipeline source text.",
                }
            },
            "required": ["dot_source"],
        },
    },
    {
        "name": "attractor_run_pipeline",
        "description": "Execute an Attractor software factory pipeline against a target workspace directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dot_source": {
                    "type": "string",
                    "description": "The complete Graphviz DOT pipeline source text.",
                },
                "workspace": {
                    "type": "string",
                    "description": "Path to the target codebase/workspace on disk.",
                },
                "auto_approve": {
                    "type": "boolean",
                    "description": "Auto-approve human gates (Lights-out mode). Default is true for MCP.",
                    "default": True,
                },
            },
            "required": ["dot_source"],
        },
    },
    {
        "name": "attractor_list_presets",
        "description": "List all built-in Software Factory blueprints and preset pipelines.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "attractor_parse_graph",
        "description": "Parse a Graphviz DOT source and return a structured AST of nodes, shapes, goal, and edges.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dot_source": {
                    "type": "string",
                    "description": "The Graphviz DOT source string.",
                }
            },
            "required": ["dot_source"],
        },
    },
    {
        "name": "attractor_export_diagram",
        "description": "Convert an Attractor DOT software factory pipeline into GitHub-compatible Mermaid flowchart or terminal ASCII diagram.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dot_source": {
                    "type": "string",
                    "description": "The Graphviz DOT source string.",
                },
                "format": {
                    "type": "string",
                    "enum": ["mermaid", "ascii"],
                    "description": "Output format: 'mermaid' (default) or 'ascii'.",
                    "default": "mermaid",
                },
            },
            "required": ["dot_source"],
        },
    },
]


class AttractorMcpServer:
    """Standard Model Context Protocol (MCP) JSON-RPC 2.0 handler."""

    def __init__(self) -> None:
        self.runner = PipelineRunner(interviewer=AutoApproveInterviewer())

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOL_DEFINITIONS},
            }

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                result_text = self._call_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "attractor_list_presets":
            return json.dumps(
                [
                    {
                        "id": p["id"],
                        "title": p["title"],
                        "summary": p["summary"],
                        "default_workspace": p.get("default_workspace"),
                        "dot_source": p["dot_source"],
                    }
                    for p in PRESET_PIPELINES
                ],
                indent=2,
            )

        if name == "attractor_parse_graph":
            dot_source = str(args.get("dot_source", ""))
            graph = parse_dot(dot_source)
            return json.dumps(
                {
                    "name": graph.name,
                    "goal": graph.goal,
                    "label": graph.label,
                    "nodes": [
                        {
                            "id": n.id,
                            "label": n.label,
                            "shape": n.shape,
                            "type": n.node_type,
                            "prompt": n.prompt,
                            "goal_gate": n.goal_gate,
                        }
                        for n in graph.nodes.values()
                    ],
                    "edges": [
                        {
                            "from": e.from_node,
                            "to": e.to_node,
                            "label": e.label,
                            "condition": e.condition,
                            "weight": e.weight,
                        }
                        for e in graph.edges
                    ],
                },
                indent=2,
            )

        if name == "attractor_lint_pipeline":
            dot_source = str(args.get("dot_source", ""))
            try:
                graph, diags = self.runner.prepare(dot_source)
            except DotParseError as pe:
                return json.dumps(
                    {
                        "valid": False,
                        "error": str(pe),
                        "diagnostics": [
                            {"rule": "syntax", "severity": "error", "message": str(pe)}
                        ],
                    },
                    indent=2,
                )

            has_errors = any(d.severity == Severity.ERROR for d in diags)
            return json.dumps(
                {
                    "valid": not has_errors,
                    "graph_name": graph.name,
                    "goal": graph.goal,
                    "diagnostics": [d.model_dump() for d in diags],
                },
                indent=2,
            )

        if name == "attractor_run_pipeline":
            dot_source = str(args.get("dot_source", ""))
            workspace = args.get("workspace")
            auto_approve = bool(args.get("auto_approve", True))
            ws_path = Path(workspace).resolve() if workspace else None

            interviewer = AutoApproveInterviewer() if auto_approve else None
            runner = PipelineRunner(interviewer=interviewer, workspace=ws_path)
            outcome = runner.run(dot_source)

            return json.dumps(
                {
                    "status": str(outcome.status.value),
                    "notes": outcome.notes,
                    "completed_nodes": outcome.completed_nodes,
                    "failure_reason": outcome.failure_reason,
                    "workspace": str(ws_path) if ws_path else None,
                },
                indent=2,
            )

        if name == "attractor_export_diagram":
            dot_source = str(args.get("dot_source", ""))
            fmt = str(args.get("format", "mermaid")).lower()
            if fmt == "ascii":
                return dot_to_ascii(dot_source)
            return dot_to_mermaid(dot_source)

        raise ValueError(f"Unknown tool name: {name}")

    def run_stdio(self) -> None:
        """Run standard I/O JSON-RPC loop for MCP clients."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


def main() -> None:
    server = AttractorMcpServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
