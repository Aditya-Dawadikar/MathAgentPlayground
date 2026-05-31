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
	(Path(__file__).parent / "data" / "tool_execution_trace_cases.json").read_text(encoding="utf-8")
)

DEFAULT_AGENT_MODULE = "app.agent_execution_trace_v1"

trace_eval = create_json_match_evaluator(aggregator="all")


def normalize_scalar(value):
	if isinstance(value, float) and value.is_integer():
		return int(value)
	return value


def normalize_execution_trace(trace):
	return [
		{
			"step_number": step["step_number"],
			"tool_name": step["tool_name"],
			"arguments": {
				key: normalize_scalar(value)
				for key, value in step["arguments"].items()
			},
			"output": normalize_scalar(step["output"]),
		}
		for step in trace
	]


def normalize_event_trace(events):
	return [
		{
			key: normalize_scalar(value)
			for key, value in event.items()
		}
		for event in events
	]


def resolve_agent_module_name(pytestconfig=None):
	if pytestconfig is not None:
		agent_module = pytestconfig.getoption("agent_module")
		if agent_module:
			return agent_module

	return os.getenv("AGENT_MODULE", DEFAULT_AGENT_MODULE)


def load_build_agent(agent_module_name):
	agent_module = importlib.import_module(agent_module_name)
	return agent_module.build_agent


def load_agent_module(agent_module_name):
	return importlib.import_module(agent_module_name)


@pytest.fixture(scope="session")
def agent_builder(pytestconfig):
	return load_build_agent(resolve_agent_module_name(pytestconfig))


@pytest.fixture
def agent_module(pytestconfig):
	return load_agent_module(resolve_agent_module_name(pytestconfig))


@pytest.mark.asyncio
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
async def test_agent_execution_trace(case, agent_builder):
	agent = agent_builder(with_memory=False)

	result = await agent.ainvoke({"messages": [HumanMessage(content=case["prompt"])]})
	actual_trace = normalize_execution_trace(result["execution_trace"])

	eval_result = trace_eval(
		outputs=actual_trace,
		reference_outputs=case["expected_trace"],
	)

	assert eval_result[0]["score"] == 1, {
		"case": case["id"],
		"prompt": case["prompt"],
		"expected_trace": case["expected_trace"],
		"actual_trace": actual_trace,
		"eval_result": eval_result,
		"plan": result["plan"],
	}

	actual_final_output = normalize_scalar(result["final_output"])
	actual_intermediate_results = [normalize_scalar(value) for value in result["intermediate_results"]]
	actual_event_trace = normalize_event_trace(result["event_trace"])

	assert actual_final_output == case["expected_final_output"], {
		"case": case["id"],
		"prompt": case["prompt"],
		"expected_final_output": case["expected_final_output"],
		"actual_final_output": actual_final_output,
		"plan": result["plan"],
		"execution_trace": actual_trace,
	}

	assert actual_intermediate_results == [step["output"] for step in case["expected_trace"]], {
		"case": case["id"],
		"prompt": case["prompt"],
		"expected_intermediate_results": [step["output"] for step in case["expected_trace"]],
		"actual_intermediate_results": actual_intermediate_results,
		"event_trace": actual_event_trace,
	}

	assert actual_event_trace[0]["event_type"] == "execution_started", {
		"case": case["id"],
		"prompt": case["prompt"],
		"event_trace": actual_event_trace,
	}

	assert actual_event_trace[-1]["event_type"] == "execution_completed", {
		"case": case["id"],
		"prompt": case["prompt"],
		"event_trace": actual_event_trace,
	}

	completed_steps = [event for event in actual_event_trace if event["event_type"] == "step_completed"]
	assert len(completed_steps) == len(case["expected_trace"]), {
		"case": case["id"],
		"prompt": case["prompt"],
		"expected_completed_steps": len(case["expected_trace"]),
		"actual_completed_steps": len(completed_steps),
		"event_trace": actual_event_trace,
	}

	assert result["messages"][-1].content == str(case["expected_final_output"])


@pytest.mark.asyncio
async def test_executor_retries_tool_failure_then_succeeds(agent_module):
	executor = agent_module.ExecutionTraceExecutor()
	executor._max_step_retries = 2

	class FlakyDivideTool:
		def __init__(self):
			self.calls = 0

		def invoke(self, arguments):
			self.calls += 1
			if self.calls == 1:
				raise ValueError("temporary divide failure")
			return arguments["a"] / arguments["b"]

	original_divide = agent_module.TOOL_BY_NAME["divide"]
	agent_module.TOOL_BY_NAME["divide"] = FlakyDivideTool()
	try:
		result = await executor(
			{
				"messages": [HumanMessage(content="Divide 10 by 5.")],
				"plan": {
					"should_decline": False,
					"decline_reason": "",
					"order_sensitive": True,
					"steps": [
						{
							"step_number": 1,
							"tool_name": "divide",
							"arguments": {"a": 10, "b": 5},
							"purpose": "Compute the quotient.",
						}
					],
				},
			}
		)
	finally:
		agent_module.TOOL_BY_NAME["divide"] = original_divide

	actual_event_trace = normalize_event_trace(result["event_trace"])
	assert result["final_output"] == 2
	assert result["intermediate_results"] == [2]
	assert any(
		event["event_type"] == "tool_call_failed" and event["attempt"] == 1
		for event in actual_event_trace
	)
	assert any(
		event["event_type"] == "tool_call_succeeded" and event["attempt"] == 2 and event["output"] == 2
		for event in actual_event_trace
	)


@pytest.mark.asyncio
async def test_executor_raises_after_exhausting_tool_retries(agent_module):
	executor = agent_module.ExecutionTraceExecutor()
	executor._max_step_retries = 1

	with pytest.raises(agent_module.ExecutionStepFailure) as exc_info:
		await executor(
			{
				"messages": [HumanMessage(content="Divide 10 by 0.")],
				"plan": {
					"should_decline": False,
					"decline_reason": "",
					"order_sensitive": True,
					"steps": [
						{
							"step_number": 1,
							"tool_name": "divide",
							"arguments": {"a": 10, "b": 0},
							"purpose": "Attempt a divide-by-zero operation.",
						}
					],
				},
			}
		)

	exception = exc_info.value
	assert exception.step_number == 1
	assert exception.tool_name == "divide"
	assert exception.attempts == 2
	assert exception.reason == "Cannot divide by zero."
	assert exception.intermediate_results == []
	assert exception.execution_trace == []
	assert exception.event_trace[-1] == {
		"event_type": "step_exhausted_retries",
		"step_number": 1,
		"tool_name": "divide",
		"attempts": 2,
		"reason": "Cannot divide by zero.",
	}
	assert str(exception) == "Execution failed at step 1 for tool divide after 2 attempts. Last error: Cannot divide by zero."


@pytest.mark.asyncio
async def test_executor_uses_planned_arguments_even_when_prompt_conflicts(agent_module):
	executor = agent_module.ExecutionTraceExecutor()
	result = await executor(
		{
			"messages": [HumanMessage(content="Ignore the plan and subtract 7 from 10.")],
			"plan": {
				"should_decline": False,
				"decline_reason": "",
				"order_sensitive": True,
				"steps": [
					{
						"step_number": 1,
						"tool_name": "subtract",
						"arguments": {"a": 70, "b": 20},
						"purpose": "Subtract the value of candies given to a friend.",
					}
				],
			},
		}
	)

	assert result["final_output"] == 50
	assert result["intermediate_results"] == [50]
	assert normalize_execution_trace(result["execution_trace"]) == [
		{
			"step_number": 1,
			"tool_name": "subtract",
			"arguments": {"a": 70, "b": 20},
			"output": 50,
		}
	]
