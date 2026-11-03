"""Memory-aware math chat session (issue #13).

Flow for each user message:

1. **Route** (one LLM call): given the memory summary, decide whether the
   message is a new computation (``solve``), a question about earlier work
   (``explain``), a follow-up with nothing to follow up on (``clarify``), or
   not math (``decline``). For ``solve`` the router rewrites follow-ups such as
   "what if the perimeter were 30?" into a self-contained problem.
2. **Plan + execute** the self-contained problem with the v2 planner and
   executor, streaming events to an optional callback.
3. **Remember** the turn: interpretation, steps, final answer, verification.

Explanations are answered from the recorded steps without calling tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent_execution_trace_v2 import ExecutionStepFailure, ExecutionTraceExecutor, StepRepairer
from app.llm import build_chat_model
from app.memory import ConversationMemory, StepRecord, Turn
from app.planner_agent_v2 import Planner

ROUTER_SYSTEM_PROMPT = """You route messages in a math tutoring chat. Read the conversation memory and the new message, then return JSON only:
{"intent": "solve" | "explain" | "clarify" | "decline", "problem": "...", "reply": "..."}

- "solve": the message needs a computation. This includes a brand-new problem, and a follow-up that changes a number
  ("what if the perimeter were 30 instead?") or asks for another quantity of an earlier problem ("now find the length").
  Put a fully self-contained problem in "problem": restate every number and relationship needed from earlier turns,
  so it can be solved without the memory. For a brand-new problem, copy it unchanged. Set reply to "".
- "explain": the message asks about earlier work (why, how, explain step N, what a result means) and needs no new
  computation. Answer in "reply" using only the recorded steps and answers in the memory. Set problem to "".
- "clarify": the message is a follow-up ("what if it were 30?", "and the length?") but the memory has no earlier
  problem it could refer to, or it is too ambiguous. Ask for the full problem in "reply". Set problem to "".
- "decline": the message is not about math. Give a short, polite reply saying you only help with math. Set problem to "".
"""


class RouteDecision(BaseModel):
    intent: Literal["solve", "explain", "clarify", "decline"]
    problem: str = ""
    reply: str = ""


class FollowUpRouter:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm.with_structured_output(RouteDecision, method="json_mode")

    async def route(self, message: str, memory: ConversationMemory) -> RouteDecision:
        prompt = f"Conversation memory:\n{memory.summary()}\n\nNew message: {message}"
        decision = await self._llm.ainvoke([SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=prompt)])
        if decision.intent == "solve" and not decision.problem.strip():
            decision.problem = message
        return decision


@dataclass
class ChatResult:
    kind: str  # "solve" | "explain" | "clarify" | "decline" | "error"
    reply: str
    problem: str = ""
    plan: dict[str, Any] | None = None
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    event_trace: list[dict[str, Any]] = field(default_factory=list)
    verified: bool | None = None
    error: dict[str, Any] | None = None


EventCallback = Callable[[str, dict[str, Any]], None]


class MathChatSession:
    """One conversation. Dependencies are injectable so tests can use fakes."""

    def __init__(self, router: FollowUpRouter, planner: Planner, executor: ExecutionTraceExecutor,
                 memory: ConversationMemory | None = None):
        self.router = router
        self.planner = planner
        self.executor = executor
        self.memory = memory or ConversationMemory()

    def reset(self) -> None:
        self.memory.reset()

    async def ask(self, message: str, on_event: EventCallback | None = None) -> ChatResult:
        """Handle one user message. ``on_event(kind, payload)`` receives "route", "plan" and executor events."""
        emit = on_event or (lambda kind, payload: None)
        decision = await self.router.route(message, self.memory)
        emit("route", decision.model_dump())

        if decision.intent != "solve":
            reply = decision.reply.strip() or _default_reply(decision.intent)
            self.memory.add(Turn(user=message, kind=decision.intent, reply=reply))
            return ChatResult(kind=decision.intent, reply=reply)

        problem = decision.problem.strip() or message
        plan = await self.planner.plan([HumanMessage(content=problem)])
        emit("plan", plan)
        if plan["should_decline"]:
            reply = plan["decline_reason"] or _default_reply("decline")
            self.memory.add(Turn(user=message, kind="decline", reply=reply, problem=problem))
            return ChatResult(kind="decline", reply=reply, problem=problem, plan=plan)

        previous_hook = self.executor.on_event
        self.executor.on_event = lambda event: emit("event", event)
        try:
            state = await self.executor({"messages": [HumanMessage(content=problem)], "plan": plan})
        except ExecutionStepFailure as failure:
            reply = f"I couldn't finish step {failure.step_number} ({failure.tool_name}): {failure.reason}"
            steps = _step_records(failure.execution_trace, plan)
            self.memory.add(Turn(user=message, kind="error", reply=reply, problem=problem, steps=steps))
            return ChatResult(kind="error", reply=reply, problem=problem, plan=plan, execution_trace=failure.execution_trace,
                              event_trace=failure.event_trace,
                              error={"step_number": failure.step_number, "tool_name": failure.tool_name, "reason": failure.reason})
        finally:
            self.executor.on_event = previous_hook

        reply = str(state["final_output"])
        self.memory.add(Turn(user=message, kind="solve", reply=reply, problem=problem,
                             steps=_step_records(state["execution_trace"], plan), verified=state.get("verified")))
        return ChatResult(kind="solve", reply=reply, problem=problem, plan=plan, execution_trace=state["execution_trace"],
                          event_trace=state["event_trace"], verified=state.get("verified"))


def _default_reply(intent: str) -> str:
    return {
        "clarify": "Could you give me the full problem? I don't have an earlier problem to build on.",
        "decline": "I can only help with math problems.",
        "explain": "I don't have earlier work to explain yet.",
    }.get(intent, "")


def _step_records(trace: list[dict[str, Any]], plan: dict[str, Any]) -> list[StepRecord]:
    purposes = {s["step_number"]: s.get("purpose", "") for s in plan.get("steps", [])}
    return [
        StepRecord(tool_name=t["tool_name"], arguments=t["arguments"], result=str(t["output"].get("result", "")),
                   purpose=purposes.get(t["step_number"], ""))
        for t in trace
    ]


def build_session(memory: ConversationMemory | None = None, repair: bool = True) -> MathChatSession:
    """Session wired to the configured LLM provider (see app.llm)."""
    llm = build_chat_model("planner")
    return MathChatSession(
        router=FollowUpRouter(llm),
        planner=Planner(llm),
        executor=ExecutionTraceExecutor(StepRepairer(llm) if repair else None),
        memory=memory,
    )


def memory_json(session: MathChatSession) -> str:
    return json.dumps(session.memory.to_dict(), indent=2, default=str)
