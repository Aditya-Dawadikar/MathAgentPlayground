import json
import os
import re
from datetime import datetime
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


PLANNER_SYSTEM_PROMPT = """You are a planner agent.

You are not an execution agent. Do not call tools. Do not solve the problem directly.
Your job is to return a structured tool-use plan only.

Available tool names:
- add
- subtract
- multiply
- divide
- power
- get_current_datetime

Return JSON only and follow this exact contract:
{
  "should_decline": false,
  "decline_reason": "",
  "order_sensitive": true,
  "steps": [
	{
	  "step_number": 1,
	  "tool_name": "add",
	  "arguments": {"a": 1, "b": 2},
	  "purpose": "short explanation"
	}
  ]
}

Rules:
- If the query is not a math problem or should not be handled as a math planning task, set should_decline=true, provide a short decline_reason, set steps=[], and set order_sensitive=true.
- If the query is a math planning task, set should_decline=false and decline_reason="".
- Decline requests such as creative writing, summarization, explanation, general advice, brainstorming, or other non-math tasks.
- Decline vague requests that do not contain a concrete math problem to solve with tools.
- When declining, do not invent any steps or tool calls.
- A valid decline response looks like this:
	{
		"should_decline": true,
		"decline_reason": "The query is not a math problem.",
		"order_sensitive": true,
		"steps": []
	}
- Examples that should be declined:
	- "Write a short poem about winter rain."
	- "Explain what photosynthesis is in simple terms."
	- "Help me decide whether I should move to a new city."
	- "Summarize the main idea of this meeting note."
- Use only these tool names: add, subtract, multiply, divide, power, get_current_datetime.
- For add, subtract, multiply, and divide, arguments must use keys a and b.
- For power, arguments must use keys base and exponent.
- For get_current_datetime, arguments must be {}.
- Use numeric values in arguments, including computed intermediate values when a later step depends on an earlier one.
- Never use placeholders such as result_of_step_1. Substitute the numeric value directly.
- Set order_sensitive=false only when independent subcomputations can be performed in any order without changing correctness.
- Every step must include step_number, tool_name, arguments, and purpose.
"""


EXECUTOR_SYSTEM_PROMPT = """Executor decisions are derived deterministically from the normalized plan."""


class PlannerStep(BaseModel):
	step_number: int = Field(description="1-based position of the step in the plan.")
	tool_name: Literal["add", "subtract", "multiply", "divide", "power", "get_current_datetime"]
	arguments: dict[str, float | str] = Field(description="Exact arguments for the chosen tool.")
	purpose: str = Field(description="Short explanation of why this step is needed.")


class PlannerResponse(BaseModel):
	should_decline: bool
	decline_reason: str
	order_sensitive: bool
	steps: list[PlannerStep]


class ExecutorToolCall(BaseModel):
	model_config = ConfigDict(extra="forbid")

	tool_name: Literal["add", "subtract", "multiply", "divide", "power", "get_current_datetime"]
	arguments: dict[str, float | str]
	reasoning: str = Field(description="Short explanation of how the current step uses the accumulated context.")


class ExecutionStepFailure(RuntimeError):
	def __init__(
		self,
		*,
		step_number: int,
		tool_name: str,
		attempts: int,
		reason: str,
		event_trace: list[dict[str, Any]],
		execution_trace: list[dict[str, Any]],
		intermediate_results: list[Any],
	):
		self.step_number = step_number
		self.tool_name = tool_name
		self.attempts = attempts
		self.reason = reason
		self.event_trace = event_trace
		self.execution_trace = execution_trace
		self.intermediate_results = intermediate_results
		super().__init__(
			f"Execution failed at step {step_number} for tool {tool_name} after {attempts} attempts. Last error: {reason}"
		)


RESULT_REFERENCE_PATTERN = re.compile(r"^result_of_step_(\d+)$")


def _coerce_argument_value(value: float | str, step_results: list[float | None]) -> float | str:
	if isinstance(value, (int, float)):
		return float(value)

	match = RESULT_REFERENCE_PATTERN.fullmatch(value.strip())
	if match:
		step_index = int(match.group(1)) - 1
		if step_index < 0 or step_index >= len(step_results) or step_results[step_index] is None:
			raise RuntimeError(f"Unresolvable step result reference: {value}")
		return float(step_results[step_index])

	try:
		return float(value)
	except ValueError:
		return value


def _compute_step_result(tool_name: str, arguments: dict[str, float | str]) -> float | None:
	if tool_name == "add":
		return float(arguments["a"]) + float(arguments["b"])
	if tool_name == "subtract":
		return float(arguments["a"]) - float(arguments["b"])
	if tool_name == "multiply":
		return float(arguments["a"]) * float(arguments["b"])
	if tool_name == "divide":
		return float(arguments["a"]) / float(arguments["b"])
	if tool_name == "power":
		return float(arguments["base"]) ** float(arguments["exponent"])
	return None


def _normalize_plan(plan: dict[str, Any]) -> PlannerResponse:
	normalized_steps: list[dict[str, Any]] = []
	step_results: list[float | None] = []

	for step in plan.get("steps", []):
		normalized_arguments = {
			key: _coerce_argument_value(value, step_results)
			for key, value in step.get("arguments", {}).items()
		}
		normalized_step = {
			"step_number": step["step_number"],
			"tool_name": step["tool_name"],
			"arguments": normalized_arguments,
			"purpose": step["purpose"],
		}
		normalized_steps.append(normalized_step)
		step_results.append(_compute_step_result(step["tool_name"], normalized_arguments))

	normalized_plan = {
		"should_decline": bool(plan.get("should_decline", False)),
		"decline_reason": str(plan.get("decline_reason", "")),
		"order_sensitive": bool(plan.get("order_sensitive", True)),
		"steps": normalized_steps,
	}
	return PlannerResponse.model_validate(normalized_plan)


def _normalize_scalar(value: Any) -> Any:
	if isinstance(value, float) and value.is_integer():
		return int(value)
	return value


@tool
def add(a: float, b: float) -> float:
	"""Add two numeric values.

	Use this for addition requests such as plus, add, total, or sum.
	Format the tool input as {"a": <first number>, "b": <second number>}.
	Arguments must be named a and b.
	"""
	return a + b


@tool
def subtract(a: float, b: float) -> float:
	"""Subtract b from a.

	Use this for subtraction requests such as subtract, minus, difference, or take away.
	Format the tool input as {"a": <starting number>, "b": <number to subtract>}.
	Arguments must be named a and b, where the operation is a - b.
	"""
	return a - b


@tool
def multiply(a: float, b: float) -> float:
	"""Multiply two numeric values.

	Use this for multiplication requests such as multiply, times, or product.
	Format the tool input as {"a": <first factor>, "b": <second factor>}.
	Arguments must be named a and b.
	"""
	return a * b


@tool
def divide(a: float, b: float) -> float:
	"""Divide a by b.

	Use this for division requests such as divide or quotient.
	Format the tool input as {"a": <dividend>, "b": <divisor>}.
	Arguments must be named a and b, where the operation is a / b.
	"""
	if b == 0:
		raise ValueError("Cannot divide by zero.")
	return a / b


@tool
def power(base: float, exponent: float) -> float:
	"""Raise base to the power of exponent.

	Use this for exponentiation requests such as power, raised to, squared, or cubed.
	Format the tool input as {"base": <base number>, "exponent": <power>}.
	Arguments must be named base and exponent. Do not use a and b for this tool.
	"""
	return base ** exponent


@tool
def get_current_datetime() -> str:
	"""Return the current date and time.

	Use this when the user asks for the current date, current time, or both.
	Format the tool input as {} because this tool takes no arguments.
	This tool takes no arguments.
	"""
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [add, subtract, multiply, divide, power, get_current_datetime]
TOOL_BY_NAME = {tool.name: tool for tool in TOOLS}


class ExecutionTraceState(TypedDict, total=False):
	messages: Annotated[list[BaseMessage], add_messages]
	plan: dict[str, Any]
	planner_system_prompt: str
	executor_system_prompt: str
	intermediate_results: list[Any]
	event_trace: list[dict[str, Any]]
	execution_trace: list[dict[str, Any]]
	final_output: Any


class ExecutionTracePlanner:
	def __init__(self):
		planner_llm = ChatOllama(
			model=os.getenv("OLLAMA_MODEL_2", "llama3.1:8b"),
			base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
			temperature=0,
		)
		self._planner = planner_llm.with_structured_output(
			PlannerResponse,
			method="json_mode",
			include_raw=True,
		)

	async def __call__(self, state: ExecutionTraceState) -> ExecutionTraceState:
		messages = state.get("messages", [])
		prompt_messages: list[BaseMessage] = [SystemMessage(content=PLANNER_SYSTEM_PROMPT), *messages]
		result = await self._planner.ainvoke(prompt_messages)

		parsed = result.get("parsed")
		if parsed is not None:
			normalized_plan = _normalize_plan(parsed.model_dump())
		else:
			parsing_error = result.get("parsing_error")
			raw_response = result.get("raw")
			raw_content = getattr(raw_response, "content", raw_response)
			try:
				raw_plan = json.loads(raw_content)
				normalized_plan = _normalize_plan(raw_plan)
			except Exception as exc:
				raise RuntimeError(
					"Planner returned an invalid structured response. "
					f"parsing_error={parsing_error!r}, raw_content={raw_content!r}"
				) from exc

		return {
			"plan": normalized_plan.model_dump(),
			"planner_system_prompt": PLANNER_SYSTEM_PROMPT,
		}

def _coerce_final_output(value: Any) -> Any:
	if isinstance(value, str):
		stripped = value.strip()
		try:
			parsed = json.loads(stripped)
		except json.JSONDecodeError:
			try:
				return _normalize_scalar(float(stripped))
			except ValueError:
				return stripped
		return _normalize_scalar(parsed)
	return _normalize_scalar(value)


def _serialize_event_value(value: Any) -> Any:
	if isinstance(value, dict):
		return {key: _serialize_event_value(item) for key, item in value.items()}
	if isinstance(value, list):
		return [_serialize_event_value(item) for item in value]
	return _normalize_scalar(value)


def _append_event(event_trace: list[dict[str, Any]], event_type: str, **payload: Any) -> None:
	event = {"event_type": event_type, **{key: _serialize_event_value(value) for key, value in payload.items()}}
	event_trace.append(event)


def _resolve_step_arguments(arguments: dict[str, Any], intermediate_results: list[Any]) -> dict[str, Any]:
	resolved_arguments = {}
	for key, value in arguments.items():
		resolved_value = _coerce_argument_value(value, intermediate_results)
		resolved_arguments[key] = _normalize_scalar(resolved_value)
	return resolved_arguments


def _validate_step_decision(
	decision: ExecutorToolCall,
	expected_tool_name: str,
	expected_arguments: dict[str, Any],
) -> tuple[bool, str]:
	if decision.tool_name != expected_tool_name:
		return False, f"Expected tool {expected_tool_name} but executor selected {decision.tool_name}."

	normalized_actual = {
		key: _normalize_scalar(value)
		for key, value in decision.arguments.items()
	}
	if normalized_actual != expected_arguments:
		return False, f"Expected arguments {expected_arguments} but executor selected {normalized_actual}."

	return True, ""


class ExecutionTraceExecutor:
	def __init__(self):
		self._max_step_retries = int(os.getenv("EXECUTOR_MAX_STEP_RETRIES", "2"))

	async def __call__(self, state: ExecutionTraceState) -> ExecutionTraceState:
		plan = state["plan"]
		messages = state.get("messages", [])
		user_query = "\n".join(
			str(message.content)
			for message in messages
			if not isinstance(message, SystemMessage)
		)
		event_trace: list[dict[str, Any]] = []
		execution_trace: list[dict[str, Any]] = []
		intermediate_results: list[Any] = []

		_append_event(
			event_trace,
			"execution_started",
			step_count=len(plan["steps"]),
			order_sensitive=plan["order_sensitive"],
		)

		if plan["should_decline"]:
			final_output = plan["decline_reason"]
			_append_event(event_trace, "execution_declined", reason=final_output)
			return {
				"executor_system_prompt": EXECUTOR_SYSTEM_PROMPT,
				"intermediate_results": [],
				"event_trace": event_trace,
				"execution_trace": [],
				"final_output": final_output,
				"messages": [AIMessage(content=str(final_output))],
			}

		for step in plan["steps"]:
			step_number = step["step_number"]
			expected_arguments = _resolve_step_arguments(step["arguments"], intermediate_results)
			decision = ExecutorToolCall(
				tool_name=step["tool_name"],
				arguments=expected_arguments,
				reasoning=step["purpose"],
			)
			_append_event(
				event_trace,
				"step_started",
				step_number=step_number,
				tool_name=step["tool_name"],
				expected_arguments=expected_arguments,
				purpose=step["purpose"],
			)

			step_completed = False
			last_error = ""

			for attempt in range(1, self._max_step_retries + 2):
				_append_event(event_trace, "step_attempt_started", step_number=step_number, attempt=attempt)
				tool_instance = TOOL_BY_NAME[decision.tool_name]
				_append_event(
					event_trace,
					"tool_call_started",
					step_number=step_number,
					attempt=attempt,
					tool_name=decision.tool_name,
					arguments=expected_arguments,
				)
				try:
					raw_output = tool_instance.invoke(expected_arguments)
				except Exception as exc:
					last_error = str(exc)
					_append_event(
						event_trace,
						"tool_call_failed",
						step_number=step_number,
						attempt=attempt,
						tool_name=decision.tool_name,
						reason=last_error,
					)
					continue

				step_output = _normalize_scalar(raw_output)
				intermediate_results.append(step_output)
				trace_entry = {
					"step_number": step_number,
					"tool_name": decision.tool_name,
					"arguments": expected_arguments,
					"output": step_output,
				}
				execution_trace.append(trace_entry)
				_append_event(
					event_trace,
					"tool_call_succeeded",
					step_number=step_number,
					attempt=attempt,
					tool_name=decision.tool_name,
					arguments=expected_arguments,
					output=step_output,
				)
				_append_event(
					event_trace,
					"step_completed",
					step_number=step_number,
					attempt=attempt,
					output=step_output,
				)
				step_completed = True
				break

			if not step_completed:
				_append_event(
					event_trace,
					"step_exhausted_retries",
					step_number=step_number,
					tool_name=step["tool_name"],
					attempts=self._max_step_retries + 1,
					reason=last_error,
				)
				normalized_events = json.loads(json.dumps(event_trace, ensure_ascii=True))
				normalized_trace = json.loads(json.dumps(execution_trace, ensure_ascii=True))
				normalized_intermediate_results = json.loads(json.dumps(intermediate_results, ensure_ascii=True))
				raise ExecutionStepFailure(
					step_number=step_number,
					tool_name=step["tool_name"],
					attempts=self._max_step_retries + 1,
					reason=last_error,
					event_trace=normalized_events,
					execution_trace=normalized_trace,
					intermediate_results=normalized_intermediate_results,
				)

		final_output = execution_trace[-1]["output"] if execution_trace else ""
		normalized_trace = json.loads(json.dumps(execution_trace, ensure_ascii=True))
		normalized_events = json.loads(json.dumps(event_trace, ensure_ascii=True))
		normalized_intermediate_results = json.loads(json.dumps(intermediate_results, ensure_ascii=True))
		normalized_output = json.loads(json.dumps(_coerce_final_output(final_output), ensure_ascii=True))
		_append_event(normalized_events, "execution_completed", final_output=normalized_output)

		return {
			"executor_system_prompt": EXECUTOR_SYSTEM_PROMPT,
			"intermediate_results": normalized_intermediate_results,
			"event_trace": normalized_events,
			"execution_trace": normalized_trace,
			"final_output": normalized_output,
			"messages": [AIMessage(content=str(normalized_output))],
		}


def build_agent(with_memory: bool = False):
	del with_memory
	graph = StateGraph(ExecutionTraceState)
	graph.add_node("planner", ExecutionTracePlanner())
	graph.add_node("executor", ExecutionTraceExecutor())
	graph.add_edge(START, "planner")
	graph.add_edge("planner", "executor")
	graph.add_edge("executor", END)
	return graph.compile()
