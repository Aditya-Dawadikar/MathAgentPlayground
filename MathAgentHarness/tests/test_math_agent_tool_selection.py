"""LLM eval: does the ReAct agent pick the right math tool with equivalent arguments?

Default agent: app.agent_tool_call_correctness_v4 (override with --agent-module).
Arguments are compared semantically, so "2x+3=11" matches "2*x + 3 = 11".
Sad paths: non-math prompts must not call math tools, and tool errors
(divide by zero, no solution) must be explained in the final answer.
"""

import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from tests.math_eval_utils import contains_any, tool_call_matches

pytestmark = pytest.mark.llm

TEST_CASES = json.loads((Path(__file__).parent / "data" / "math_tool_selection_cases.json").read_text(encoding="utf-8"))
DEFAULT_AGENT_MODULE = "app.agent_tool_call_correctness_v4"


def extract_tool_calls(result):
    return [
        {"name": call["name"], "args": call["args"]}
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]


@pytest.fixture(scope="session")
def agent(pytestconfig):
    import importlib

    module_name = pytestconfig.getoption("agent_module") or os.getenv("AGENT_MODULE", DEFAULT_AGENT_MODULE)
    return importlib.import_module(module_name).build_agent(with_memory=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
async def test_math_tool_selection(case, agent):
    result = await agent.ainvoke({"messages": [HumanMessage(content=case["prompt"])]})
    calls = extract_tool_calls(result)
    final_answer = result["messages"][-1].content
    details = {"case": case["id"], "prompt": case["prompt"], "calls": calls, "final_answer": final_answer}

    if not case["expected"]:
        assert calls == [], details
        return

    assert calls, details
    assert tool_call_matches(calls[0], case["expected"]), {**details, "expected": case["expected"]}
    if case.get("final_answer_contains"):
        assert contains_any(final_answer, case["final_answer_contains"]), details
