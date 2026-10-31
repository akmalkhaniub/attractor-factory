"""Comprehensive verification of the Attractor Specification (Section 11 Parity Matrix & Smoke Test)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from attractor import (
    Checkpoint,
    Context,
    HandlerRegistry,
    Node,
    Outcome,
    PipelineRunner,
    QueueInterviewer,
    Severity,
    StageStatus,
    ValidationError,
    parse_dot,
    run_pipeline,
    select_edge,
    validate,
    validate_or_raise,
)
from attractor.api import app


def test_parse_simple_linear_pipeline() -> None:
    dot = """
    digraph Simple {
        graph [goal="Run tests and report", label="Simple Pipeline"]
        rankdir=LR

        start [shape=Mdiamond, label="Start"]
        exit  [shape=Msquare, label="Exit"]

        run_tests [label="Run Tests", prompt="Run the test suite for: $goal"]
        report    [label="Report", prompt="Summarize the test results"]

        start -> run_tests -> report -> exit
    }
    """
    graph = parse_dot(dot)
    assert graph.name == "Simple"
    assert graph.goal == "Run tests and report"
    assert graph.label == "Simple Pipeline"
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3


def test_parse_multiline_node_attributes_and_subgraphs() -> None:
    dot = """
    digraph Multi {
        // Line comment
        /* Block comment */
        start [shape=Mdiamond]
        subgraph cluster_loop {
            label = "Loop A"
            node [thread_id="loop-a", timeout="900s"]
            review_gate [
                shape=hexagon,
                label="Review Changes",
                type="wait.human"
            ]
        }
        exit [shape=Msquare]
        start -> review_gate -> exit
    }
    """
    graph = parse_dot(dot)
    gate = graph.nodes["review_gate"]
    assert gate.shape == "hexagon"
    assert gate.node_type == "wait.human"
    assert gate.thread_id == "loop-a"
    assert "loop-a" in gate.classes


def test_validation_missing_start_and_exit_and_orphan() -> None:
    # Missing start node -> error
    no_start = parse_dot("digraph G { a [shape=box, prompt=\"hi\"]; exit [shape=Msquare]; a -> exit }")
    diags = validate(no_start)
    assert any(d.rule == "start_node" and d.severity == Severity.ERROR for d in diags)

    # Missing exit node -> error
    no_exit = parse_dot("digraph G { start [shape=Mdiamond]; a [shape=box, prompt=\"hi\"]; start -> a }")
    diags2 = validate(no_exit)
    assert any(d.rule == "terminal_node" and d.severity == Severity.ERROR for d in diags2)

    # Unreachable orphan node -> error
    orphan = parse_dot(
        """
        digraph G {
            start [shape=Mdiamond]
            exit  [shape=Msquare]
            orphan_node [shape=box, prompt="isolated"]
            start -> exit
        }
        """
    )
    diags3 = validate(orphan)
    assert any(d.rule == "reachability" and d.node_id == "orphan_node" for d in diags3)
    with pytest.raises(ValidationError):
        validate_or_raise(orphan)


def test_conditional_branching_and_edge_selection_priority() -> None:
    dot = """
    digraph Branch {
        graph [goal="Validate branching"]
        start [shape=Mdiamond]
        step1 [shape=box, prompt="Do step 1"]
        gate  [shape=diamond, label="Check"]
        path_a [shape=box, prompt="Path A"]
        path_b [shape=box, prompt="Path B"]
        exit  [shape=Msquare]

        start -> step1 -> gate
        gate -> path_a [condition="outcome=success && context.flag=alpha"]
        gate -> path_b [weight=10]
        path_a -> exit
        path_b -> exit
    }
    """
    ctx = Context({"flag": "alpha"})
    out = run_pipeline(dot, context=ctx)
    assert out.status == StageStatus.SUCCESS
    assert "path_a" in out.completed_nodes
    assert "path_b" not in out.completed_nodes

    # Weight & lexical tiebreak on unconditional edges
    graph = parse_dot(
        """
        digraph Tie {
            start [shape=Mdiamond]
            b_node [shape=box, prompt="B"]
            a_node [shape=box, prompt="A"]
            c_node [shape=box, prompt="C"]
            exit [shape=Msquare]
            start -> b_node [weight=5]
            start -> a_node [weight=5]
            start -> c_node [weight=1]
            a_node -> exit
            b_node -> exit
            c_node -> exit
        }
        """
    )
    chosen = select_edge(graph.nodes["start"], Outcome(status=StageStatus.SUCCESS), Context(), graph)
    assert chosen is not None
    assert chosen.to_node == "a_node"


def test_retry_on_failure_and_goal_gate_enforcement() -> None:
    attempts = {"count": 0}

    def flaky_backend(node: Node, prompt: str, context: Context) -> str | Outcome:
        if node.id == "flaky":
            attempts["count"] += 1
            if attempts["count"] < 3:
                return Outcome(status=StageStatus.RETRY, failure_reason="transient rate limit")
        return f"Completed {node.id}"

    dot = """
    digraph RetryTest {
        graph [goal="Test retry logic"]
        start [shape=Mdiamond]
        flaky [shape=box, prompt="Try task", max_retries=2, goal_gate=true]
        exit  [shape=Msquare]
        start -> flaky -> exit
    }
    """
    out = run_pipeline(dot, llm_callback=flaky_backend)
    assert out.status == StageStatus.SUCCESS
    assert attempts["count"] == 3


def test_goal_gate_blocks_exit_when_unsatisfied() -> None:
    def failing_gate_backend(node: Node, prompt: str, context: Context) -> str | Outcome:
        if node.id == "must_pass":
            return Outcome(status=StageStatus.FAIL, failure_reason="tests failed")
        return "ok"

    dot = """
    digraph GateFail {
        start [shape=Mdiamond]
        must_pass [shape=box, prompt="Critical step", goal_gate=true]
        fallback  [shape=box, prompt="Fallback path"]
        exit      [shape=Msquare]

        start -> must_pass
        must_pass -> fallback [condition="outcome=fail"]
        must_pass -> exit     [condition="outcome=success"]
        fallback -> exit
    }
    """
    out = run_pipeline(dot, llm_callback=failing_gate_backend)
    assert out.status == StageStatus.FAIL
    assert "Goal gate" in out.failure_reason


def test_wait_human_interviewer_routing() -> None:
    dot = """
    digraph Review {
        start [shape=Mdiamond]
        review_gate [shape=hexagon, label="Review Changes", type="wait.human"]
        ship_it [shape=box, prompt="Deploy"]
        fixes   [shape=box, prompt="Fix issues"]
        exit    [shape=Msquare]

        start -> review_gate
        review_gate -> ship_it [label="[A] Approve"]
        review_gate -> fixes   [label="[F] Fix"]
        ship_it -> exit
        fixes -> exit
    }
    """
    interviewer = QueueInterviewer(["F"])
    out = run_pipeline(dot, interviewer=interviewer)
    assert out.status == StageStatus.SUCCESS
    assert "fixes" in out.completed_nodes
    assert "ship_it" not in out.completed_nodes


def test_stylesheet_specificity_and_variable_expansion() -> None:
    dot = """
    digraph StylesheetGraph {
        graph [
            goal="Build feature X",
            model_stylesheet="
                * { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }
                .code { llm_model: claude-opus-4-6; llm_provider: anthropic; }
                #critical_review { llm_model: gpt-5.2; llm_provider: openai; reasoning_effort: high; }
            "
        ]
        start [shape=Mdiamond]
        plan [label="Plan", class="planning", prompt="Plan $goal"]
        implement [label="Implement", class="code", prompt="Code $goal"]
        critical_review [label="Critical Review", class="code", prompt="Review $goal"]
        override_node [label="Override", class="code", llm_model="gemini-2.5-pro", prompt="Custom"]
        exit [shape=Msquare]

        start -> plan -> implement -> critical_review -> override_node -> exit
    }
    """
    runner = PipelineRunner()
    graph, diags = runner.prepare(dot)
    assert not [d for d in diags if d.severity == Severity.ERROR]
    assert graph.nodes["plan"].llm_model == "claude-sonnet-4-5"
    assert graph.nodes["implement"].llm_model == "claude-opus-4-6"
    assert graph.nodes["critical_review"].llm_model == "gpt-5.2"
    assert graph.nodes["override_node"].llm_model == "gemini-2.5-pro"
    assert graph.nodes["plan"].prompt == "Plan Build feature X"


def test_parallel_fan_out_and_fan_in_and_checkpoint_resume(tmp_path: Path) -> None:
    dot = """
    digraph ParallelPipeline {
        graph [goal="Test parallel fan-out and fan-in"]
        start [shape=Mdiamond]
        fan_out [shape=component, label="Fan Out"]
        branch_a [shape=box, prompt="Branch A"]
        branch_b [shape=box, prompt="Branch B"]
        fan_in [shape=tripleoctagon, label="Fan In"]
        exit [shape=Msquare]

        start -> fan_out
        fan_out -> branch_a
        fan_out -> branch_b
        branch_a -> fan_in
        branch_b -> fan_in
        fan_in -> exit
    }
    """
    runner = PipelineRunner(logs_root=tmp_path)
    out = runner.run(dot)
    assert out.status == StageStatus.SUCCESS
    assert "fan_in" in out.completed_nodes

    cp = Checkpoint.load(tmp_path / "checkpoint.json")
    assert cp.current_node == "exit"
    assert cp.context.get("parallel.fan_in.best_id") in ("branch_a", "branch_b")


def test_section_11_13_integration_smoke_test(tmp_path: Path) -> None:
    dot = """
    digraph test_pipeline {
        graph [goal="Create a hello world Python script"]

        start       [shape=Mdiamond]
        plan        [shape=box, prompt="Plan how to create a hello world script for: $goal"]
        implement   [shape=box, prompt="Write the code based on the plan", goal_gate=true]
        review      [shape=box, prompt="Review the code for correctness"]
        done        [shape=Msquare]

        start -> plan
        plan -> implement
        implement -> review [condition="outcome=success"]
        implement -> plan   [condition="outcome=fail", label="Retry"]
        review -> done      [condition="outcome=success"]
        review -> implement [condition="outcome=fail", label="Fix"]
    }
    """
    graph = parse_dot(dot)
    assert graph.goal == "Create a hello world Python script"
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 6

    lint_results = validate(graph)
    assert not [d for d in lint_results if d.severity == Severity.ERROR]

    ctx = Context()
    out = run_pipeline(
        graph,
        context=ctx,
        llm_callback=lambda n, p, c: f"Real callback output for {n.id}: {p}",
        logs_root=tmp_path,
    )
    assert out.status == StageStatus.SUCCESS
    assert "implement" in out.completed_nodes

    for stage in ("plan", "implement", "review"):
        for fname in ("prompt.md", "response.md", "status.json"):
            assert (tmp_path / stage / fname).exists()

    cp = Checkpoint.load(tmp_path / "checkpoint.json")
    assert cp.current_node == "done"
    assert "plan" in cp.completed_nodes
    assert "implement" in cp.completed_nodes
    assert "review" in cp.completed_nodes


def test_fastapi_server_endpoints() -> None:
    with TestClient(app) as client:
        presets_resp = client.get("/api/presets")
        assert presets_resp.status_code == 200
        presets = presets_resp.json()["presets"]
        assert len(presets) >= 3

        val_resp = client.post(
            "/api/pipelines/validate",
            json={"dot_source": presets[0]["dot_source"]},
        )
        assert val_resp.status_code == 200
        assert val_resp.json()["valid"] is True

        run_resp = client.post(
            "/api/pipelines",
            json={
                "dot_source": presets[2]["dot_source"],
                "auto_approve": True,
                "step_delay_ms": 0,
            },
        )
        assert run_resp.status_code == 202
        pid = run_resp.json()["pipeline_id"]

        status_resp = client.get(f"/api/pipelines/{pid}")
        assert status_resp.status_code == 200
