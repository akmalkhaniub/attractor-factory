"""Tests for the Layer 2 Coding Agent Loop & Workspace Environment."""

from __future__ import annotations

from pathlib import Path

from attractor import (
    CodingAgentSession,
    Context,
    LocalExecutionEnvironment,
    Node,
    PipelineRunner,
    QueueInterviewer,
    StageStatus,
    WorkspaceCodingBackend,
    create_standard_tool_registry,
    run_pipeline,
)


def test_local_execution_environment_file_ops(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    assert env.working_directory() == tmp_path

    # write_file
    res = env.write_file("sub/hello.txt", "line 1\nline 2\nline 3\n")
    assert "Successfully wrote" in res
    assert (tmp_path / "sub" / "hello.txt").exists()

    # read_file line numbering
    read_out = env.read_file("sub/hello.txt")
    assert "1 | line 1" in read_out
    assert "2 | line 2" in read_out

    # edit_file
    edit_res = env.edit_file("sub/hello.txt", "line 2", "LINE TWO REPLACED")
    assert "Successfully made 1 replacement" in edit_res
    updated = (tmp_path / "sub" / "hello.txt").read_text(encoding="utf-8")
    assert "LINE TWO REPLACED" in updated

    # grep & glob
    grep_res = env.grep("REPLACED")
    assert "sub/hello.txt" in grep_res or "sub\\hello.txt" in grep_res

    glob_res = env.glob("**/*.txt")
    assert any("hello.txt" in g for g in glob_res)


def test_tool_registry_execution(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    tools = create_standard_tool_registry()

    # write via tool
    w_out = tools.execute("write_file", {"file_path": "code.py", "content": "print('hello')\n"}, env)
    assert "Successfully wrote" in w_out

    # list_directory via tool
    l_out = tools.execute("list_directory", {"path": "."}, env)
    assert "code.py" in l_out

    # shell command via tool
    s_out = tools.execute("shell", {"command": "python code.py"}, env)
    assert "hello" in s_out
    assert "exit code: 0" in s_out


def test_workspace_coding_backend_autonomous_execution(tmp_path: Path) -> None:
    backend = WorkspaceCodingBackend(workspace_root=tmp_path)
    ctx = Context({"graph.goal": "Create verified webhook verifier"})

    # 1. Plan node
    plan_node = Node(id="plan", attrs={"label": "Architecture Plan", "prompt": "Draft architectural blueprint for: $goal"})
    plan_out = backend.run(plan_node, "Draft architectural blueprint", ctx)
    assert (tmp_path / "SPECIFICATION.md").exists()
    assert "Architecture Blueprint Established" in str(plan_out)

    # 2. Implement node
    impl_node = Node(id="implement", attrs={"label": "Synthesize Code", "prompt": "Implement code and tests"})
    impl_out = backend.run(impl_node, "Implement code and tests", ctx)
    assert (tmp_path / "src" / "service.py").exists()
    assert (tmp_path / "tests" / "test_service.py").exists()
    assert "Implementation Code Synthesized" in str(impl_out)


def test_end_to_end_factory_pipeline_on_target_workspace(tmp_path: Path) -> None:
    """End-to-end pipeline operating strictly inside a target workspace."""
    dot = """
    digraph TargetFactory {
        graph [goal="Build idempotent webhook service with cryptographic verification"]
        rankdir=LR

        start     [shape=Mdiamond]
        plan      [shape=box, prompt="Design an implementation blueprint for: $goal"]
        implement [shape=box, prompt="Write production code and unit tests for: $goal", goal_gate=true]
        verify    [shape=parallelogram, label="Run Tests", command="python -m pytest tests -q"]
        gate      [shape=diamond, label="Tests Passing?"]
        review    [shape=hexagon, label="Human Approval Gate"]
        exit      [shape=Msquare]

        start -> plan -> implement -> verify -> gate
        gate -> review [condition="outcome=success"]
        gate -> plan   [condition="outcome!=success"]
        review -> exit [label="[A] Approve & Release"]
    }
    """
    interviewer = QueueInterviewer(["A"])
    runner = PipelineRunner(
        interviewer=interviewer,
        workspace=tmp_path,
    )
    outcome = runner.run(dot)
    assert outcome.status == StageStatus.SUCCESS
    assert "implement" in outcome.completed_nodes
    assert "verify" in outcome.completed_nodes

    # Verify that the target workspace now contains real working code and git history
    assert (tmp_path / "SPECIFICATION.md").exists()
    assert (tmp_path / "src" / "service.py").exists()
    assert (tmp_path / "tests" / "test_service.py").exists()
    assert (tmp_path / ".git").exists()
