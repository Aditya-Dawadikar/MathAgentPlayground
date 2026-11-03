"""Deterministic tests for chat memory, the chat session and the CLI (no LLM).

The router and planner are replaced by fakes; the real executor and math
tools run, so these tests exercise the whole turn except the LLM calls.
"""

import io

import pytest

from app import cli
from app.agent_execution_trace_v2 import ExecutionTraceExecutor
from app.chat_session import ChatResult, MathChatSession, RouteDecision
from app.memory import ConversationMemory, StepRecord, Turn


# ---------------------------------------------------------------- fakes


class FakeRouter:
    """Returns scripted decisions and records the memory summary it was shown."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.seen = []

    async def route(self, message, memory):
        self.seen.append((message, memory.summary()))
        return self.decisions.pop(0)


class FakePlanner:
    """Maps a problem statement to a hand-written plan."""

    def __init__(self, plans):
        self.plans = plans
        self.problems = []

    async def plan(self, messages):
        problem = messages[-1].content
        self.problems.append(problem)
        steps = self.plans.get(problem)
        if steps is None:
            return {"should_decline": True, "decline_reason": "The query is not a math problem.", "order_sensitive": True, "steps": []}
        return {"should_decline": False, "decline_reason": "", "order_sensitive": True,
                "steps": [{"step_number": i, "tool_name": t, "arguments": a, "purpose": p} for i, (t, a, p) in enumerate(steps, 1)]}


RECTANGLE_26 = "A rectangle's length is 3 more than its width and its perimeter is 26. What is its area?"
RECTANGLE_30 = "A rectangle's length is 3 more than its width and its perimeter is 30. What is its area?"
PLANS = {
    RECTANGLE_26: [("solve_equation", {"equation": "2w + 2(w + 3) = 26"}, "find the width"),
                   ("multiply", {"a": "$step_1", "b": "($step_1) + 3"}, "area = width x length")],
    RECTANGLE_30: [("solve_equation", {"equation": "2w + 2(w + 3) = 30"}, "find the width"),
                   ("multiply", {"a": "$step_1", "b": "($step_1) + 3"}, "area = width x length")],
    "Divide 1 by 0.": [("divide", {"a": 1, "b": 0}, "divide")],
}


def session(*decisions, memory=None):
    return MathChatSession(FakeRouter(*decisions), FakePlanner(PLANS), ExecutionTraceExecutor(), memory)


def solve(problem):
    return RouteDecision(intent="solve", problem=problem)


# ---------------------------------------------------------------- memory


def test_memory_summary_and_bounds():
    memory = ConversationMemory(max_detailed=2, max_condensed=2)
    assert memory.summary() == "(no earlier turns)" and not memory.has_work
    for i in range(6):
        memory.add(Turn(user=f"problem {i}", kind="solve", reply=str(i), problem=f"problem {i}",
                        steps=[StepRecord("add", {"a": i, "b": 0}, str(i))], verified=True))
    assert len(memory) == 4  # 2 detailed + 2 condensed, oldest dropped
    summary = memory.summary()
    assert "problem 0" not in summary and "problem 1" not in summary
    assert "Older turns (condensed):" in summary and "- problem 2 -> 2 (verified)" in summary
    assert "step 1: add(a=5, b=0) -> 5" in summary
    memory.reset()
    assert len(memory) == 0


# ---------------------------------------------------------------- session: happy paths


@pytest.mark.asyncio
async def test_solve_then_follow_up_uses_memory():
    chat = session(solve(RECTANGLE_26), solve(RECTANGLE_30),
                   RouteDecision(intent="explain", reply="The length is 3 more than the width, so I added 3."))
    first = await chat.ask(RECTANGLE_26)
    assert first.kind == "solve" and first.reply == "40"

    follow_up = await chat.ask("what if the perimeter were 30 instead?")
    assert follow_up.kind == "solve" and follow_up.reply == "54"
    assert follow_up.problem == RECTANGLE_30
    # The router saw the first turn's steps when interpreting the follow-up.
    _, summary = chat.router.seen[1]
    assert "2w + 2(w + 3) = 26" in summary and "final answer: 40" in summary

    explain = await chat.ask("why did you add 3?")
    assert explain.kind == "explain" and "3 more" in explain.reply
    assert chat.planner.problems == [RECTANGLE_26, RECTANGLE_30]  # explain did not plan or run tools
    assert [t.kind for t in chat.memory.turns] == ["solve", "solve", "explain"]


@pytest.mark.asyncio
async def test_events_are_streamed_and_hook_restored():
    chat = session(solve(RECTANGLE_26))
    seen = []
    await chat.ask(RECTANGLE_26, on_event=lambda kind, payload: seen.append((kind, payload.get("event_type"))))
    kinds = [k for k, _ in seen]
    assert kinds[:2] == ["route", "plan"]
    assert ("event", "tool_call_succeeded") in seen
    assert chat.executor.on_event is None


# ---------------------------------------------------------------- session: sad paths


@pytest.mark.asyncio
async def test_clarify_and_decline_do_not_run_tools():
    chat = session(RouteDecision(intent="clarify", reply="Could you give me the full problem?"),
                   RouteDecision(intent="decline", reply=""))
    clarify = await chat.ask("what if it were 30 instead?")
    decline = await chat.ask("write a poem")
    assert clarify.kind == "clarify" and "full problem" in clarify.reply
    assert decline.kind == "decline" and decline.reply == "I can only help with math problems."
    assert chat.planner.problems == []


@pytest.mark.asyncio
async def test_planner_decline_is_reported():
    chat = session(solve("Tell me a joke."))
    result = await chat.ask("Tell me a joke.")
    assert result.kind == "decline"
    assert chat.memory.turns[-1].kind == "decline"


@pytest.mark.asyncio
async def test_failed_step_becomes_error_turn():
    chat = session(solve("Divide 1 by 0."))
    result = await chat.ask("Divide 1 by 0.")
    assert result.kind == "error"
    assert result.error["tool_name"] == "divide" and "math_error" in result.error["reason"]
    assert chat.memory.turns[-1].kind == "error" and chat.memory.has_work


# ---------------------------------------------------------------- CLI


@pytest.mark.asyncio
async def test_cli_renders_plan_steps_and_answer():
    out = io.StringIO()
    renderer = cli.Renderer(out=out)
    await cli.run_chat(session(solve(RECTANGLE_26), solve(RECTANGLE_30)), renderer,
                       questions=[RECTANGLE_26, "what if the perimeter were 30 instead?"])
    text = out.getvalue()
    assert "Plan (2 steps)" in text
    assert "Step 1  solve_equation(equation='2w + 2(w + 3) = 26')" in text
    assert "-> w = 5" in text
    assert "Answer: 40" in text
    assert f"Interpreting as: {RECTANGLE_30}" in text
    assert "Answer: 54" in text


@pytest.mark.asyncio
async def test_cli_commands_and_interactive_loop():
    chat = session(solve(RECTANGLE_26))
    out = io.StringIO()
    stream = io.StringIO("/help\n" + RECTANGLE_26 + "\n/history\n/trace off\n/trace maybe\n/reset\n/history\n/bogus\n/exit\nnever read\n")
    await cli.run_chat(chat, cli.Renderer(out=out), stream=stream)
    text = out.getvalue()
    assert "Commands:" in text
    assert "final answer: 40" in text  # /history before reset
    assert "Step-by-step trace off." in text and "Usage: /trace on|off" in text
    assert "Memory cleared." in text and "(no earlier turns)" in text
    assert "Unknown command /bogus" in text
    assert len(chat.memory) == 0


@pytest.mark.asyncio
async def test_cli_trace_off_prints_only_answers_and_survives_errors():
    class BrokenSession:
        memory = ConversationMemory()

        async def ask(self, message, on_event=None):
            raise ConnectionError("model offline")

    out = io.StringIO()
    await cli.run_chat(session(solve(RECTANGLE_26)), cli.Renderer(out=out, trace=False), questions=[RECTANGLE_26])
    assert "Plan" not in out.getvalue() and "Answer: 40" in out.getvalue()

    out = io.StringIO()
    await cli.run_chat(BrokenSession(), cli.Renderer(out=out), questions=["Solve 2x = 4"])
    assert "something went wrong talking to the model: ConnectionError: model offline" in out.getvalue()


def test_cli_renders_error_and_decline_results():
    out = io.StringIO()
    renderer = cli.Renderer(out=out)
    renderer.result(ChatResult(kind="error", reply="I couldn't finish step 1 (divide): math_error"))
    renderer.result(ChatResult(kind="decline", reply="I can only help with math problems."))
    assert "Sorry - I couldn't finish step 1" in out.getvalue()
    assert "I can only help with math problems." in out.getvalue()


def test_cli_main_with_injected_session(capsys):
    assert cli.main(["-q", RECTANGLE_26, "--no-trace"], session_factory=lambda repair: session(solve(RECTANGLE_26))) == 0
    assert "Answer: 40" in capsys.readouterr().out


# ---------------------------------------------------------------- API


def test_api_session_solve_and_delete(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    monkeypatch.setattr(main, "build_session", lambda: session(solve(RECTANGLE_26), solve(RECTANGLE_30)))
    monkeypatch.setattr(main, "_chat_sessions", {})
    with TestClient(main.app) as client:
        first = client.post("/sessions/s1/solve", json={"message": RECTANGLE_26}).json()
        follow_up = client.post("/sessions/s1/solve", json={"message": "what if the perimeter were 30 instead?"}).json()
        assert (first["kind"], first["reply"]) == ("solve", "40")
        assert (follow_up["reply"], follow_up["problem"]) == ("54", RECTANGLE_30)
        assert client.delete("/sessions/s1").json()["cleared"] is True
        assert client.delete("/sessions/s1").json()["cleared"] is False
