"""LLM eval: multi-turn chat with memory (issue #13).

Each case is a short conversation run through a fresh MathChatSession. Every
turn checks the routed kind (solve / explain / clarify / decline) and, for
solve turns, the final answer. A "/reset" turn clears memory mid-conversation.
"""

import json
from pathlib import Path

import pytest

from app.chat_session import build_session
from tests.math_eval_utils import answers_match, contains_any

pytestmark = pytest.mark.llm

TEST_CASES = json.loads((Path(__file__).parent / "data" / "math_followup_cases.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
async def test_math_chat_followups(case):
    session = build_session()
    for index, turn in enumerate(case["turns"], start=1):
        if turn["user"] == "/reset":
            session.reset()
            continue
        result = await session.ask(turn["user"])
        details = {"case": case["id"], "turn": index, "user": turn["user"], "kind": result.kind, "reply": result.reply,
                   "interpreted_as": result.problem, "memory": session.memory.summary()}
        assert result.kind == turn["kind"], details
        if "expected_answers" in turn:
            assert answers_match(result.reply, turn["expected_answers"]), details
        if "reply_contains" in turn:
            assert contains_any(result.reply, turn["reply_contains"]), details
