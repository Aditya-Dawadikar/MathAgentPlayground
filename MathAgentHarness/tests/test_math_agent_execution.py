"""LLM eval: does the planner -> executor graph reach the right final answer?

Default agent: app.agent_execution_trace_v2 (override with --agent-module).
Plans may differ between runs, so only the final answer is checked
(semantically and order-insensitively), plus that verification never fails.
Sad paths: decline non-math prompts, and report "no solution" correctly.
"""

import importlib
import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from tests.math_eval_utils import answer_assignments, answers_match, contains_any, equivalent

pytestmark = pytest.mark.llm

TEST_CASES = json.loads((Path(__file__).parent / "data" / "math_execution_cases.json").read_text(encoding="utf-8"))
DEFAULT_AGENT_MODULE = "app.agent_execution_trace_v2"


@pytest.fixture(scope="session")
def agent(pytestconfig):
    module_name = pytestconfig.getoption("agent_module") or os.getenv("AGENT_MODULE", DEFAULT_AGENT_MODULE)
    return importlib.import_module(module_name).build_agent(with_memory=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
async def test_math_execution(case, agent):
    result = await agent.ainvoke({"messages": [HumanMessage(content=case["prompt"])]})
    final_output = result["final_output"]
    details = {
        "case": case["id"],
        "prompt": case["prompt"],
        "final_output": final_output,
        "plan": result["plan"],
        "execution_trace": [{k: s[k] for k in ("tool_name", "arguments")} for s in result["execution_trace"]],
    }

    if case.get("expect_decline"):
        assert result["plan"]["should_decline"] is True, details
        assert result["execution_trace"] == [], details
        return

    assert result["plan"]["should_decline"] is False, details
    assert result.get("verified") is not False, {**details, "reason": "a verify_solution step failed"}
    if "expected_answers" in case:
        assert answers_match(final_output, case["expected_answers"]), details
    if "expected_assignments" in case:
        actual = answer_assignments(final_output)
        assert all(equivalent(actual.get(k), v) for k, v in case["expected_assignments"].items()), details
    if "expected_contains" in case:
        assert contains_any(final_output, case["expected_contains"]), details
