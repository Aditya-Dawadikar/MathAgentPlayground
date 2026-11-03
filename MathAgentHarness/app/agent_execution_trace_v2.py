"""Planner -> executor graph over the full math toolset (issue #11).

The executor runs the plan deterministically:
- ``$step_N...`` references in arguments are resolved against earlier tool outputs.
- Tools return result/error envelopes. A tool exception is retried as-is
  (it may be transient); an error envelope is deterministic, so the step is
  handed to an optional LLM *repairer* that sees the error message and proposes
  corrected arguments. Without a repairer the step fails immediately.
- The final answer is the result of the last non-verification step; the
  outcome of a trailing ``verify_solution`` step is reported as ``verified``.

State/response shape matches v1 (plan, intermediate_results, event_trace,
execution_trace, final_output) plus ``verified``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.llm import build_chat_model
from app.math_tools import TOOLS_BY_NAME, describe_tools, suggest_tools
from app.planner_agent_v2 import DEFAULT_SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT
from app.planner_agent_v2 import REFERENCE, Planner

EXECUTOR_SYSTEM_PROMPT = """Executor decisions are derived deterministically from the plan; failed steps may be repaired by an LLM."""

REPAIR_SYSTEM_PROMPT = f"""A step in a math tool plan failed. Propose a corrected call for this one step.

Available tools:
{describe_tools()}

Rules:
- Read the error message and fix the cause (wrong tool, wrong argument names, malformed expression).
- tool_name must be exactly one of the tool names listed above (for example "add", never a heading like "arithmetic").
- Keep the same intent as the original step. You may use $step_N references to earlier results.
- Return JSON only: {{"tool_name": "...", "arguments": {{...}}, "reasoning": "short explanation"}}
"""

# Tests patch this mapping to inject flaky tools.
TOOL_BY_NAME: dict[str, Any] = dict(TOOLS_BY_NAME)
DEFAULT_REFERENCE_FIELD = "exact"
REFERENCE_PATH = re.compile(r"\.([A-Za-z_]\w*)|\[(-?\d+)\]")
VERIFY_TOOL = "verify_solution"


class ExecutionStepFailure(RuntimeError):
    def __init__(self, *, step_number: int, tool_name: str, attempts: int, reason: str,
                 event_trace: list[dict[str, Any]], execution_trace: list[dict[str, Any]], intermediate_results: list[Any]):
        self.step_number = step_number
        self.tool_name = tool_name
        self.attempts = attempts
        self.reason = reason
        self.event_trace = event_trace
        self.execution_trace = execution_trace
        self.intermediate_results = intermediate_results
        super().__init__(f"Execution failed at step {step_number} for tool {tool_name} after {attempts} attempts. Last error: {reason}")


class ReferenceResolutionError(ValueError):
    pass


# --------------------------------------------------------------------------- references


def _walk(value: Any, path: str, reference: str) -> Any:
    for token in REFERENCE_PATH.findall(path):
        field, index = token
        try:
            value = value[field] if field else value[int(index)]
        except (KeyError, IndexError, TypeError) as exc:
            available = f"fields: {', '.join(value)}" if isinstance(value, dict) else f"{type(value).__name__} of length {len(value)}" if isinstance(value, list) else repr(value)
            raise ReferenceResolutionError(f"cannot resolve {reference}: no {field or f'[{index}]'} ({available})") from exc
    return value


def _lookup(match: re.Match, outputs: list[dict[str, Any]]) -> Any:
    step = int(match.group(1) or match.group(3))
    if step < 1 or step > len(outputs):
        raise ReferenceResolutionError(f"{match.group(0)} refers to a step that has not run yet")
    output = outputs[step - 1]
    path = match.group(2) or ""
    if not path:
        return output.get(DEFAULT_REFERENCE_FIELD, output.get("result"))
    return _walk(output, path, match.group(0))


def resolve_references(value: Any, outputs: list[dict[str, Any]]) -> Any:
    """Replace $step_N references with earlier outputs (whole values keep their type)."""
    if isinstance(value, dict):
        return {k: resolve_references(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_references(v, outputs) for v in value]
    if not isinstance(value, str):
        return value
    whole = REFERENCE.fullmatch(value.strip())
    if whole:
        return _lookup(whole, outputs)

    def inline(match: re.Match) -> str:
        found = _lookup(match, outputs)
        if isinstance(found, (list, dict)):
            raise ReferenceResolutionError(f"{match.group(0)} is a list/object and cannot be used inside an expression")
        text = str(found)
        source, start, end = match.string, match.start(), match.end()
        already_wrapped = source[:start].rstrip().endswith("(") and source[end:].lstrip().startswith(")")
        # Simple non-negative atoms (5, x) need no parentheses; anything else keeps precedence safe.
        return text if already_wrapped or re.fullmatch(r"[A-Za-z0-9_.]+", text) else f"({text})"

    return REFERENCE.sub(inline, value)


# --------------------------------------------------------------------------- repair


class RepairedCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class StepRepairer:
    """Asks an LLM for a corrected call after a tool returned an error envelope."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm.with_structured_output(RepairedCall, method="json_mode")

    async def repair(self, *, problem: str, step: dict[str, Any], error: dict[str, Any], outputs: list[dict[str, Any]]) -> RepairedCall:
        context = {
            "problem": problem,
            "failed_step": step,
            "error": error,
            "earlier_results": [{"step": i + 1, "result": o.get("result"), "exact": o.get("exact")} for i, o in enumerate(outputs)],
        }
        return await self._llm.ainvoke([SystemMessage(content=REPAIR_SYSTEM_PROMPT), SystemMessage(content=json.dumps(context, default=str))])


# --------------------------------------------------------------------------- graph


class ExecutionTraceState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: dict[str, Any]
    planner_system_prompt: str
    executor_system_prompt: str
    intermediate_results: list[Any]
    event_trace: list[dict[str, Any]]
    execution_trace: list[dict[str, Any]]
    final_output: Any
    verified: bool | None


class ExecutionTracePlanner:
    def __init__(self, planner: Planner | None = None):
        self._planner = planner or Planner(build_chat_model("planner"))

    async def __call__(self, state: ExecutionTraceState) -> ExecutionTraceState:
        plan = await self._planner.plan(state.get("messages", []))
        return {"plan": plan, "planner_system_prompt": PLANNER_SYSTEM_PROMPT}


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, default=str))


def _event(trace: list[dict[str, Any]], event_type: str, **payload: Any) -> None:
    trace.append({"event_type": event_type, **_jsonable(payload)})


def _final_answer(answer_step: dict[str, Any], trace: list[dict[str, Any]]) -> Any:
    """Result of the answer step, narrowed to the roots the plan chose to verify.

    If a solve step returned several solutions (w = -8 or w = 6) and the plan
    verified only some of them afterwards (w = 6, because a width is positive),
    the verified ones are the answer.
    """
    output = answer_step["output"]
    result = output.get("result", "")
    solutions = output.get("solutions")
    variable = output.get("variable")
    if not variable or not isinstance(solutions, list) or len(solutions) < 2:
        return result
    later = trace[trace.index(answer_step) + 1:]
    chosen = []
    for step in later:
        if step["tool_name"] != VERIFY_TOOL or not step["output"].get("is_solution"):
            continue
        value = (step["arguments"].get("values") or {}).get(variable)
        match = next((s for s in solutions if str(s) == str(value) or _same_number(s, value)), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    if not chosen or len(chosen) == len(solutions):
        return result
    return " or ".join(f"{variable} = {value}" for value in chosen)


def _same_number(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


class ExecutionTraceExecutor:
    def __init__(self, repairer: StepRepairer | None = None, on_event: Callable[[dict[str, Any]], None] | None = None):
        """``on_event`` is called with each event as it happens (used by the CLI to stream steps)."""
        self._max_step_retries = int(os.getenv("EXECUTOR_MAX_STEP_RETRIES", "2"))
        self._repairer = repairer
        self.on_event = on_event

    def _record(self, events: list[dict[str, Any]], event_type: str, **payload: Any) -> None:
        _event(events, event_type, **payload)
        if self.on_event is not None:
            self.on_event(events[-1])

    async def __call__(self, state: ExecutionTraceState) -> ExecutionTraceState:
        plan = state["plan"]
        problem = "\n".join(str(m.content) for m in state.get("messages", []) if not isinstance(m, SystemMessage))
        events: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        self._record(events, "execution_started", step_count=len(plan["steps"]), order_sensitive=plan["order_sensitive"])

        if plan["should_decline"]:
            self._record(events, "execution_declined", reason=plan["decline_reason"])
            return self._finish(plan["decline_reason"], None, events, trace, outputs, declined=True)

        for step in plan["steps"]:
            output = await self._run_step(step, problem, outputs, events, trace)
            outputs.append(output)

        answer_steps = [t for t in trace if t["tool_name"] != VERIFY_TOOL] or trace
        final_output = _final_answer(answer_steps[-1], trace) if answer_steps else ""
        verifications = [t["output"].get("is_solution") for t in trace if t["tool_name"] == VERIFY_TOOL]
        verified = all(verifications) if verifications else None
        return self._finish(final_output, verified, events, trace, outputs)

    async def _run_step(self, step, problem, outputs, events, trace) -> dict[str, Any]:
        step_number = step["step_number"]
        tool_name, raw_arguments = step["tool_name"], step["arguments"]
        self._record(events, "step_started", step_number=step_number, tool_name=tool_name, arguments=raw_arguments, purpose=step.get("purpose", ""))
        last_error = ""
        attempts = self._max_step_retries + 1
        for attempt in range(1, attempts + 1):
            self._record(events, "step_attempt_started", step_number=step_number, attempt=attempt)
            try:
                arguments = resolve_references(raw_arguments, outputs)
                tool = TOOL_BY_NAME.get(tool_name)
                if tool is None:
                    hint = suggest_tools(tool_name)
                    raise LookupError(f"unknown tool {tool_name!r}" + (f"; use one of: {', '.join(hint)}" if hint else ""))
                self._record(events, "tool_call_started", step_number=step_number, attempt=attempt, tool_name=tool_name, arguments=arguments)
                output = tool.invoke(arguments)
            except Exception as exc:  # transient failures and bad references
                last_error = f"{type(exc).__name__}: {exc}"
                self._record(events, "tool_call_failed", step_number=step_number, attempt=attempt, tool_name=tool_name, reason=last_error)
                if isinstance(exc, (ReferenceResolutionError, LookupError)) and not await self._try_repair(step, problem, outputs, events, {"error": "invalid_call", "message": last_error}):
                    break
                tool_name, raw_arguments = step["tool_name"], step["arguments"]
                continue
            if not isinstance(output, dict):
                output = {"result": str(output), "exact": output}
            if "error" in output:
                last_error = f"{output['error']}: {output.get('message', '')}"
                self._record(events, "tool_call_failed", step_number=step_number, attempt=attempt, tool_name=tool_name, reason=last_error)
                if attempt < attempts and await self._try_repair(step, problem, outputs, events, output):
                    tool_name, raw_arguments = step["tool_name"], step["arguments"]
                    continue
                break
            entry = {"step_number": step_number, "tool_name": tool_name, "arguments": arguments, "output": output}
            trace.append(_jsonable(entry))
            self._record(events, "tool_call_succeeded", step_number=step_number, attempt=attempt, tool_name=tool_name, arguments=arguments, output=output)
            self._record(events, "step_completed", step_number=step_number, attempt=attempt, output=output.get("exact"))
            return _jsonable(output)

        self._record(events, "step_exhausted_retries", step_number=step_number, tool_name=tool_name, attempts=attempt, reason=last_error)
        raise ExecutionStepFailure(step_number=step_number, tool_name=tool_name, attempts=attempt, reason=last_error,
                                   event_trace=_jsonable(events), execution_trace=_jsonable(trace),
                                   intermediate_results=[o.get("exact") for o in outputs])

    async def _try_repair(self, step, problem, outputs, events, error) -> bool:
        """Mutates ``step`` with a repaired call. Returns False when no repairer is configured."""
        if self._repairer is None:
            return False
        try:
            repaired = await self._repairer.repair(problem=problem, step=step, error=error, outputs=outputs)
        except Exception as exc:
            self._record(events, "step_repair_failed", step_number=step["step_number"], reason=f"{type(exc).__name__}: {exc}")
            return False
        if repaired.tool_name == step["tool_name"] and repaired.arguments == step["arguments"]:
            self._record(events, "step_repair_failed", step_number=step["step_number"], reason="the repair proposed the same failing call")
            return False
        self._record(events, "step_repaired", step_number=step["step_number"], tool_name=repaired.tool_name,
               arguments=repaired.arguments, reasoning=repaired.reasoning)
        step["tool_name"], step["arguments"] = repaired.tool_name, repaired.arguments
        return True

    def _finish(self, final_output, verified, events, trace, outputs, declined=False) -> ExecutionTraceState:
        if not declined:
            self._record(events, "execution_completed", final_output=final_output, verified=verified)
        return {
            "executor_system_prompt": EXECUTOR_SYSTEM_PROMPT,
            "intermediate_results": [o.get("exact") for o in outputs],
            "event_trace": _jsonable(events),
            "execution_trace": _jsonable(trace),
            "final_output": _jsonable(final_output),
            "verified": verified,
            "messages": [AIMessage(content=str(final_output))],
        }


def build_agent(with_memory: bool = False):
    del with_memory
    llm = build_chat_model("planner")
    repairer = StepRepairer(llm) if os.getenv("EXECUTOR_REPAIR", "1") != "0" else None
    graph = StateGraph(ExecutionTraceState)
    graph.add_node("planner", ExecutionTracePlanner(Planner(llm)))
    graph.add_node("executor", ExecutionTraceExecutor(repairer))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", END)
    return graph.compile()
