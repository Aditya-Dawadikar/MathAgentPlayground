import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from openevals.json import create_json_match_evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_CASES = json.loads(
    (Path(__file__).parent / "data" / "tool_selection_cases.json").read_text(encoding="utf-8")
)

DEFAULT_AGENT_MODULE = "app.agent_tool_call_correctness_v1"

tool_eval = create_json_match_evaluator(
    # feedback_key = "tool_selection",
    aggregator="all"
)


def coerce_scalar(value):
    if not isinstance(value, str):
        return value

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def normalize_arguments(actual_args, expected_args):
    if not expected_args:
        return {}

    normalized_args = {
        key: coerce_scalar(value)
        for key, value in actual_args.items()
    }

    if "base" in expected_args and "base" not in normalized_args and "a" in normalized_args:
        normalized_args["base"] = normalized_args.pop("a")

    return {key: normalized_args[key] for key in expected_args if key in normalized_args}


def normalize_tool_calls(actual_tool_calls, expected_tool_calls):
    if not actual_tool_calls:
        return actual_tool_calls

    normalized_calls = []

    for actual_call, expected_call in zip(actual_tool_calls, expected_tool_calls):
        normalized_calls.append(
            {
                "name": actual_call["name"],
                "args": normalize_arguments(actual_call["args"], expected_call["args"]),
            }
        )

    return normalized_calls


def extract_tool_calls(result):
    """
        Langgraph returns messages.
        Tool Calls live on AIMessage.tool_calls.
    """

    calls = []

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                calls.append({
                    "name": call["name"] if isinstance(call, dict) else call.name,
                    "args": call["args"] if isinstance(call, dict) else call.args,
                })
    
    return calls


def resolve_agent_module_name(pytestconfig=None):
    if pytestconfig is not None:
        agent_module = pytestconfig.getoption("agent_module")
        if agent_module:
            return agent_module

    return os.getenv("AGENT_MODULE", DEFAULT_AGENT_MODULE)


def load_build_agent(agent_module_name):
    agent_module = importlib.import_module(agent_module_name)
    return agent_module.build_agent


@pytest.fixture(scope="session")
def agent_builder(pytestconfig):
    return load_build_agent(resolve_agent_module_name(pytestconfig))

@pytest.mark.asyncio
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
async def test_agent_tool_selection(case, agent_builder):
    agent = agent_builder(with_memory=False)

    result = await agent.ainvoke({"messages": [HumanMessage(content=case["prompt"])]})

    print("##############################")
    print("Agent Invocation Result: ", result)


    actual_tool_calls = normalize_tool_calls(
        extract_tool_calls(result),
        [
            {
                "name": case["expected_tool"],
                "args": case["expected_arguments"],
            }
        ],
    )

    print("##############################")
    print("Actual Tool Calls: ", actual_tool_calls)

    expected_tool_calls = [
        {
            "name": case["expected_tool"],
            "args": case["expected_arguments"]
        }
    ]

    eval_result = tool_eval(
        outputs=actual_tool_calls,
        reference_outputs=expected_tool_calls,
    )

    print("##############################")
    print(eval_result)
    
    assert eval_result[0]["score"] == 1, {
        "case": case["id"],
        "prompt": case["prompt"],
        "expected": expected_tool_calls,
        "actual": actual_tool_calls,
        "eval_result": eval_result,
    }
