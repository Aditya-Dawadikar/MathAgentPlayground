"""Simple bounded conversation memory for the math chat (issue #13).

Each turn stores what the user asked, how it was interpreted, the tool steps
with their results, and the final answer. ``summary()`` renders a compact,
prompt-friendly view: the most recent turns in detail, older turns as one
line each, and anything beyond that is forgotten.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MAX_TEXT = 160
# Recent turns keep the whole problem so follow-ups on long word problems lose nothing.
MAX_PROBLEM_TEXT = 4000


def _clip(text: Any, limit: int = MAX_TEXT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_args(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in arguments.items())


@dataclass
class StepRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: str
    purpose: str = ""

    def line(self) -> str:
        return _clip(f"{self.tool_name}({_format_args(self.arguments)}) -> {self.result}", 240)


@dataclass
class Turn:
    user: str
    kind: str  # "solve" | "explain" | "clarify" | "decline" | "error"
    reply: str = ""
    problem: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    verified: bool | None = None

    def headline(self) -> str:
        if self.kind == "solve":
            check = {True: " (verified)", False: " (verification failed)"}.get(self.verified, "")
            return f"{_clip(self.problem or self.user)} -> {_clip(self.reply, 80)}{check}"
        return f"{_clip(self.user)} -> [{self.kind}] {_clip(self.reply, 80)}"


class ConversationMemory:
    """Keeps the last ``max_detailed`` turns in detail and ``max_condensed`` older ones as one-liners."""

    def __init__(self, max_detailed: int = 4, max_condensed: int = 12):
        self.max_detailed = max_detailed
        self.max_condensed = max_condensed
        self.turns: list[Turn] = []

    def __len__(self) -> int:
        return len(self.turns)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        overflow = len(self.turns) - (self.max_detailed + self.max_condensed)
        if overflow > 0:
            del self.turns[:overflow]

    def reset(self) -> None:
        self.turns.clear()

    @property
    def has_work(self) -> bool:
        """True once at least one problem has been solved (something to follow up on)."""
        return any(t.kind in ("solve", "error") for t in self.turns)

    def last_solved(self) -> Turn | None:
        return next((t for t in reversed(self.turns) if t.kind == "solve"), None)

    def summary(self) -> str:
        if not self.turns:
            return "(no earlier turns)"
        older, recent = self.turns[: -self.max_detailed], self.turns[-self.max_detailed:]
        lines = []
        if older:
            lines.append("Older turns (condensed):")
            lines.extend(f"- {t.headline()}" for t in older)
        lines.append("Recent turns (oldest first):")
        for number, turn in enumerate(recent, start=len(older) + 1):
            lines.append(f"Turn {number} user: {_clip(turn.user, MAX_PROBLEM_TEXT)}")
            if turn.kind == "solve":
                if turn.problem and turn.problem != turn.user:
                    lines.append(f"  interpreted as: {_clip(turn.problem, MAX_PROBLEM_TEXT)}")
                for index, step in enumerate(turn.steps, start=1):
                    lines.append(f"  step {index}: {step.line()}" + (f"  # {_clip(step.purpose, 80)}" if step.purpose else ""))
                check = {True: " (verified)", False: " (verification failed)"}.get(turn.verified, "")
                lines.append(f"  final answer: {_clip(turn.reply)}{check}")
            else:
                lines.append(f"  [{turn.kind}] assistant: {_clip(turn.reply)}")
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.turns]
