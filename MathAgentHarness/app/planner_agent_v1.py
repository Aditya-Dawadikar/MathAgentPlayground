import os
import json
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


DEFAULT_SYSTEM_PROMPT = """You are a planner agent.

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


class PlannerAgent:
    def __init__(self, llm: ChatOllama):
        self._planner = llm.with_structured_output(
            PlannerResponse,
            method="json_mode",
            include_raw=True,
        )

    async def ainvoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        del config
        messages = inputs.get("messages", [])
        prompt_messages: list[BaseMessage] = [SystemMessage(content=DEFAULT_SYSTEM_PROMPT), *messages]
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

        plan = normalized_plan.model_dump()
        return {
            "plan": plan,
            "messages": [AIMessage(content=json.dumps(plan, ensure_ascii=True))],
        }


def build_agent(with_memory: bool = False):
    del with_memory
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL_2", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
    return PlannerAgent(llm)