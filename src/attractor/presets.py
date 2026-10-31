"""Curated Software Factory DOT pipeline presets for interactive experimentation."""

from __future__ import annotations

from typing import Any

PRESET_PIPELINES: list[dict[str, Any]] = [
    {
        "id": "feature-factory-hitl",
        "title": "Autonomous Feature Factory (TDD + Human Gate)",
        "summary": "Plans architecture, synthesizes implementation code, runs automated verification, gates on a Human Approval node, and enforces a Goal Gate before exit.",
        "dot_source": """digraph FeatureFactory {
    graph [
        goal="Implement idempotent webhook signature verifier with replay protection",
        label="Feature Factory Pipeline",
        default_max_retries=1,
        model_stylesheet="
            * { llm_provider: gemini; llm_model: gemini-2.5-flash; }
            .code { reasoning_effort: high; }
            #security_review { reasoning_effort: high; }
        "
    ]
    rankdir=LR
    node [shape=box, timeout="900s"]

    start           [shape=Mdiamond, label="Intake Spec"]
    plan            [label="Architecture Plan", class="planning", prompt="Design an implementation blueprint for: $goal"]
    implement       [label="Synthesize Code", class="code", prompt="Write production code and unit tests for: $goal", goal_gate=true, retry_target="plan"]
    verify_tests    [shape=parallelogram, label="Run Test Suite", tool_command="python -c \\"print('14 passed, 0 failed in 0.18s')\\""]
    test_gate       [shape=diamond, label="Tests Passing?"]
    security_review [label="Independent Review", class="code", prompt="Audit the diff for replay vulnerabilities and edge cases: $goal"]
    human_approval  [shape=hexagon, label="Release Gate Approval"]
    exit            [shape=Msquare, label="Ship Verified PR"]

    start -> plan -> implement -> verify_tests -> test_gate
    test_gate -> security_review [label="Pass", condition="outcome=success"]
    test_gate -> implement       [label="Self-Heal", condition="outcome!=success"]
    security_review -> human_approval
    human_approval -> exit       [label="[A] Approve & Merge PR"]
    human_approval -> plan       [label="[R] Request Revision"]
}""",
    },
    {
        "id": "parallel-candidate-duel",
        "title": "Parallel Multi-Agent Architecture Duel (Fan-Out / Fan-In)",
        "summary": "Fans out a specification to two parallel coding branches concurrently, consolidates results at a Fan-In arbiter node, and ships the winning implementation.",
        "dot_source": """digraph ParallelCandidateFactory {
    graph [
        goal="Build high-throughput token bucket rate limiter middleware",
        label="Parallel Candidate Duel"
    ]
    rankdir=LR

    start        [shape=Mdiamond, label="Start"]
    spec_plan    [shape=box, label="Draft NLSpec", prompt="Define strict latency and concurrency invariants for: $goal"]
    fan_out      [shape=component, label="Parallel Fan-Out", join_policy="wait_all", max_parallel=2]
    candidate_a  [shape=box, label="Candidate A (Lock-Free)", prompt="Implement lock-free atomic rate limiter for: $goal"]
    candidate_b  [shape=box, label="Candidate B (Sliding Window)", prompt="Implement sliding window rate limiter for: $goal"]
    fan_in       [shape=tripleoctagon, label="Arbiter Fan-In", prompt="Select best candidate based on verification score"]
    final_review [shape=box, label="Final QA Audit", prompt="Verify selected candidate meets all invariants for: $goal", goal_gate=true, retry_target="spec_plan"]
    exit         [shape=Msquare, label="Publish Release"]

    start -> spec_plan -> fan_out
    fan_out -> candidate_a
    fan_out -> candidate_b
    candidate_a -> fan_in
    candidate_b -> fan_in
    fan_in -> final_review -> exit
}""",
    },
    {
        "id": "zero-touch-bugfix",
        "title": "Lights-Out Issue-to-PR Bugfix Factory",
        "summary": "Non-interactive dark factory pipeline: reproduces the reported issue, patches the root cause, validates regression tests, and auto-approves release.",
        "dot_source": """digraph LightsOutBugfix {
    graph [
        goal="Fix race condition in async connection pool checkout under high concurrency",
        label="Lights-Out Bugfix Factory",
        default_max_retries=2
    ]
    rankdir=LR

    start      [shape=Mdiamond, label="Issue Webhook"]
    reproduce  [shape=box, label="Reproduce Bug", prompt="Write a deterministic reproduction test for: $goal"]
    patch_code [shape=box, label="Patch Root Cause", prompt="Implement minimal thread-safe fix for: $goal", goal_gate=true, retry_target="reproduce"]
    run_check  [shape=parallelogram, label="Verify Regression Suite", tool_command="python -c \\"print('Regression suite: 42 passed in 0.12s')\\""]
    gate       [shape=diamond, label="Verified?"]
    exit       [shape=Msquare, label="Open Signed PR"]

    start -> reproduce -> patch_code -> run_check -> gate
    gate -> exit       [label="Verified", condition="outcome=success"]
    gate -> patch_code [label="Retry Fix", condition="outcome!=success"]
}""",
    },
]
