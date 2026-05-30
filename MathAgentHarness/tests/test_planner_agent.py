import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from openevals.llm import create_llm_as_judge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_CASES = json.loads(
    (Path(__file__).parent / "data" / "tool_planner_cases.json").read_text(encoding="utf-8")
)


DEFAULT_AGENT_MODULE = "app.planner_agent_v1"

CORRECTNESS_PROMPT = """
You are a helpful and precise assistant for evaluating the correctness of an agent's tool calls plan.

This is the agent's system prompt: {system_prompt}
This is the agent's input query: {inputs}
This is the agent's generated structured plan: {outputs}
This is the expected structured plan: {reference_outputs}

Your goal is to generate a score of 1 if the agent's structured plan is equivalent to the expected structured plan, and a score of 0 if it is not.

Evaluation rules:
- Respect the should_decline field.
- Respect the decline_reason field semantically, not by exact wording.
- Compare the steps field structurally.
- If order_sensitive is false, allow equivalent reordering of independent steps.
- If order_sensitive is true, require the step order to be correct.
- Tool names and arguments must match the intended plan.
"""

judge_llm = ChatOllama(
    model=os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1:8b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
)

tool_eval = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=judge_llm,
)

def resolve_agent_module_name(pytestconfig=None):
    if pytestconfig is not None:
        agent_module = pytestconfig.getoption("agent_module")
        if agent_module:
            return agent_module

    return os.getenv("AGENT_MODULE", DEFAULT_AGENT_MODULE)


def load_build_agent(agent_module_name):
    agent_module = importlib.import_module(agent_module_name)
    return agent_module.build_agent


def load_default_system_prompt(agent_module_name):
    agent_module = importlib.import_module(agent_module_name)

    if not hasattr(agent_module, "DEFAULT_SYSTEM_PROMPT"):
        raise AttributeError(f"{agent_module_name} is missing DEFAULT_SYSTEM_PROMPT")

    return agent_module.DEFAULT_SYSTEM_PROMPT


@pytest.fixture(scope="session")
def agent_builder(pytestconfig):
    return load_build_agent(resolve_agent_module_name(pytestconfig))


@pytest.fixture(scope="session")
def agent_system_prompt(pytestconfig):
    return load_default_system_prompt(resolve_agent_module_name(pytestconfig))

@pytest.mark.asyncio
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
async def test_agent_tool_planning(case, agent_builder, agent_system_prompt):
    agent = agent_builder(with_memory=False)

    result = await agent.ainvoke({"messages": [HumanMessage(content=case["prompt"])]})
    actual_plan = result["plan"]

    print("##############################")
    print("Agent Invocation Result: ", actual_plan)

    expected_plan = {
        "should_decline": case["should_decline"],
        "order_sensitive": case["order_sensitive"],
        "steps": case["expected_steps"],
        "decline_reason": case["decline_reason"],
    }

    eval_result = tool_eval(
        system_prompt=agent_system_prompt,
        inputs=case["prompt"],
        outputs=actual_plan,
        reference_outputs=expected_plan,
    )

    print("##############################")
    print(eval_result)
    
    assert eval_result["score"] is True, {
        "case": case["id"],
        "prompt": case["prompt"],
        "expected": expected_plan,
        "actual": actual_plan,
        "eval_result": eval_result,
    }
