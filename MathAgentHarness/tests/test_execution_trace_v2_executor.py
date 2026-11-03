"""Deterministic tests for the v2 executor (no LLM needed).

Plans are written by hand so the executor logic - reference resolution,
verification, retries, repair and failure reporting - is tested on its own.
"""

import pytest
from langchain_core.messages import HumanMessage

from app import agent_execution_trace_v2 as agent
from app.agent_execution_trace_v2 import ExecutionStepFailure, ExecutionTraceExecutor, RepairedCall, resolve_references


def plan(*steps, decline=None):
    return {
        "should_decline": decline is not None,
        "decline_reason": decline or "",
        "order_sensitive": True,
        "steps": [
            {"step_number": i, "tool_name": name, "arguments": args, "purpose": ""}
            for i, (name, args) in enumerate(steps, start=1)
        ],
    }


async def run(executor, the_plan, problem="test problem"):
    return await executor({"messages": [HumanMessage(content=problem)], "plan": the_plan})


# ---------------------------------------------------------------- references


def test_resolve_references_whole_path_and_inline():
    outputs = [{"exact": "4", "result": "x = 4", "solutions": [4], "vertex": [2, -1]}]
    assert resolve_references("$step_1", outputs) == "4"
    assert resolve_references("$step_1.solutions[0]", outputs) == 4
    assert resolve_references("$step_1.vertex", outputs) == [2, -1]
    assert resolve_references("2*($step_1) + 1", outputs) == "2*((4)) + 1"
    assert resolve_references({"values": {"x": "$step_1.solutions[0]"}}, outputs) == {"values": {"x": 4}}
    assert resolve_references("result_of_step_1", outputs) == "4"


@pytest.mark.parametrize("reference", ["$step_2", "$step_1.missing", "$step_1.solutions[5]", "x + $step_1.vertex"])
def test_resolve_references_errors(reference):
    outputs = [{"exact": "4", "solutions": [4], "vertex": [2, -1]}]
    with pytest.raises(agent.ReferenceResolutionError):
        resolve_references(reference, outputs)


# ---------------------------------------------------------------- happy paths


@pytest.mark.asyncio
async def test_solve_then_verify_with_reference():
    result = await run(ExecutionTraceExecutor(), plan(
        ("solve_equation", {"equation": "2x + 3 = 11"}),
        ("verify_solution", {"equation": "2x + 3 = 11", "values": {"x": "$step_1.solutions[0]"}}),
    ))
    assert result["final_output"] == "x = 4"
    assert result["verified"] is True
    assert result["intermediate_results"] == ["4", True]
    assert result["execution_trace"][1]["arguments"] == {"equation": "2x + 3 = 11", "values": {"x": 4}}
    assert result["event_trace"][0]["event_type"] == "execution_started"
    assert result["event_trace"][-1]["event_type"] == "execution_completed"


@pytest.mark.asyncio
async def test_multi_step_word_problem_with_inline_reference():
    # Rectangle: length is 3 more than width, perimeter 26 -> width 5, area 40.
    result = await run(ExecutionTraceExecutor(), plan(
        ("solve_equation", {"equation": "2w + 2(w + 3) = 26"}),
        ("multiply", {"a": "$step_1", "b": "($step_1) + 3"}),
    ))
    assert result["final_output"] == "40"
    assert result["verified"] is None


@pytest.mark.asyncio
async def test_failed_verification_is_reported():
    result = await run(ExecutionTraceExecutor(), plan(
        ("solve_quadratic", {"a": 1, "b": -5, "c": 6}),
        ("verify_solution", {"equation": "x^2 - 5x + 6 = 0", "values": {"x": 7}}),
    ))
    assert result["final_output"] == "x = 2 or x = 3"
    assert result["verified"] is False


@pytest.mark.asyncio
async def test_decline_plan():
    result = await run(ExecutionTraceExecutor(), plan(decline="The query is not a math problem."))
    assert result["final_output"] == "The query is not a math problem."
    assert result["execution_trace"] == []


# ---------------------------------------------------------------- retries and repair


@pytest.mark.asyncio
async def test_transient_tool_exception_is_retried(monkeypatch):
    class Flaky:
        calls = 0

        def invoke(self, arguments):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise ConnectionError("temporary failure")
            return {"result": "2", "exact": "2"}

    monkeypatch.setitem(agent.TOOL_BY_NAME, "divide", Flaky())
    result = await run(ExecutionTraceExecutor(), plan(("divide", {"a": 10, "b": 5})))
    assert result["final_output"] == "2"
    assert any(e["event_type"] == "tool_call_failed" and e["attempt"] == 1 for e in result["event_trace"])


class FakeRepairer:
    def __init__(self, call):
        self.call = call
        self.seen_errors = []

    async def repair(self, *, problem, step, error, outputs):
        self.seen_errors.append(error)
        return self.call


@pytest.mark.asyncio
async def test_error_envelope_is_repaired():
    repairer = FakeRepairer(RepairedCall(tool_name="solve_equation", arguments={"equation": "2x + 3 = 11"}, reasoning="fixed typo"))
    result = await run(ExecutionTraceExecutor(repairer), plan(("solve_equation", {"equation": "2x + = 11"})))
    assert result["final_output"] == "x = 4"
    assert repairer.seen_errors[0]["error"] == "parse_error"
    assert any(e["event_type"] == "step_repaired" for e in result["event_trace"])


# ---------------------------------------------------------------- sad paths


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "steps, reason_fragment",
    [
        ([("divide", {"a": 1, "b": 0})], "math_error"),
        ([("not_a_tool", {})], "unknown tool"),
        ([("add", {"a": "$step_3", "b": 1})], "has not run yet"),
        ([("solve_equation", {"equation": "x + y = 3"})], "invalid_input"),
    ],
)
async def test_failures_raise_with_partial_traces(steps, reason_fragment):
    with pytest.raises(ExecutionStepFailure) as info:
        await run(ExecutionTraceExecutor(), plan(*steps))
    assert reason_fragment in info.value.reason
    assert info.value.event_trace[-1]["event_type"] == "step_exhausted_retries"


@pytest.mark.asyncio
async def test_failure_after_earlier_success_keeps_intermediate_results():
    with pytest.raises(ExecutionStepFailure) as info:
        await run(ExecutionTraceExecutor(), plan(("add", {"a": 2, "b": 3}), ("divide", {"a": "$step_1", "b": 0})))
    assert info.value.step_number == 2
    assert info.value.intermediate_results == ["5"]
