"""Planner agent v2: plans over the full math toolset (issue #11).

Differences from v1:
- The tool catalog in the prompt is generated from the tool registry.
- Arguments can be expressions ("2x + 3 = 11") and can reference earlier
  step outputs ("$step_1.solutions[0]") instead of pre-computed numbers,
  because most algebra results cannot be computed by the planner itself.
- Equation-solving plans end with verify_solution.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import build_chat_model
from app.math_tools import TOOL_NAMES, describe_tools

PLAN_CONTRACT = """{
  "should_decline": false,
  "decline_reason": "",
  "order_sensitive": true,
  "steps": [
    {
      "step_number": 1,
      "tool_name": "solve_equation",
      "arguments": {"equation": "2x + 3 = 11"},
      "purpose": "Solve the linear equation for x."
    },
    {
      "step_number": 2,
      "tool_name": "verify_solution",
      "arguments": {"equation": "2x + 3 = 11", "values": {"x": "$step_1.solutions[0]"}},
      "purpose": "Check the solution in the original equation."
    }
  ]
}"""

DEFAULT_SYSTEM_PROMPT = f"""You are a planner agent for a math assistant that solves 10th-grade algebra, geometry and word problems.

You are not an execution agent. Do not call tools. Do not compute the answer yourself.
Your job is to return a structured tool-use plan only.

Available tools, as name(arguments) where ? marks an optional argument:
{describe_tools()}

Return JSON only and follow this exact contract:
{PLAN_CONTRACT}

Rules:
- If the query is not a math problem, set should_decline=true, give a short decline_reason, set steps=[] and order_sensitive=true.
- Decline creative writing, summarization, explanations, general advice, brainstorming and vague requests without a concrete math problem.
- Otherwise set should_decline=false and decline_reason="".
- Use only the tool names listed above, with exactly the argument names shown.
- Write expressions and equations as strings using ^ for powers, e.g. "x^2 - 5x + 6 = 0". Numbers may be JSON numbers or exact strings like "1/3".
- Points are [x, y] lists. Shape dimensions are objects such as {{"r": 3}}.
- When a step needs an earlier result, do not compute it yourself; reference it:
  "$step_1" is step 1's exact result, "$step_1.solutions[0]" the first solution, "$step_2.roots" a list field.
  References can appear inside expressions too: "2*($step_1) + 1".
- Prefer one specific tool over many arithmetic steps (solve_equation instead of manual rearranging).
- After solving an equation or system, add a final verify_solution step that plugs the solution(s) into the ORIGINAL equation.
- Set order_sensitive=false only when the steps are independent and can run in any order.
- Every step must include step_number, tool_name, arguments and purpose.
"""

# "$step_2", "$step_2.solutions[0]", "$step_1.vertex[1]" and the v1 form "result_of_step_2".
REFERENCE = re.compile(r"\$step_(\d+)((?:\.[A-Za-z_]\w*|\[\d+\])*)|\bresult_of_step_(\d+)\b")


class PlannerStep(BaseModel):
    step_number: int = Field(description="1-based position of the step in the plan.")
    tool_name: str = Field(description="One of the available tool names.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool; may contain $step_N references.")
    purpose: str = Field(default="", description="Short explanation of why this step is needed.")


class PlannerResponse(BaseModel):
    should_decline: bool = False
    decline_reason: str = ""
    order_sensitive: bool = True
    steps: list[PlannerStep] = Field(default_factory=list)


def normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate and renumber a raw plan; references are kept for the executor to resolve."""
    response = PlannerResponse.model_validate(plan)
    if response.should_decline:
        response.steps = []
    for index, step in enumerate(response.steps, start=1):
        step.step_number = index
    return response.model_dump()


def unknown_tools(plan: dict[str, Any]) -> list[str]:
    return [s["tool_name"] for s in plan["steps"] if s["tool_name"] not in TOOL_NAMES]


class Planner:
    """Turns the conversation into a normalized plan dict."""

    def __init__(self, llm: BaseChatModel, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self._llm = llm.with_structured_output(PlannerResponse, method="json_mode", include_raw=True)

    async def plan(self, messages: list[BaseMessage]) -> dict[str, Any]:
        result = await self._llm.ainvoke([SystemMessage(content=self.system_prompt), *messages])
        parsed = result.get("parsed")
        if parsed is not None:
            return normalize_plan(parsed.model_dump())
        raw = result.get("raw")
        content = getattr(raw, "content", raw)
        try:
            return normalize_plan(json.loads(content))
        except Exception as exc:
            raise RuntimeError(
                f"Planner returned an invalid structured response. parsing_error={result.get('parsing_error')!r}, raw_content={content!r}"
            ) from exc


class PlannerAgent:
    def __init__(self, planner: Planner):
        self._planner = planner

    async def ainvoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        del config
        plan = await self._planner.plan(inputs.get("messages", []))
        return {"plan": plan, "messages": [AIMessage(content=json.dumps(plan, ensure_ascii=True))]}


def build_agent(with_memory: bool = False):
    del with_memory
    return PlannerAgent(Planner(build_chat_model("planner")))
