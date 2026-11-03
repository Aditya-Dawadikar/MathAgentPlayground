"""Interactive math chat in the terminal (issue #13).

    python -m app.cli                      # interactive chat
    python -m app.cli -q "Solve 2x + 3 = 11" -q "what if it were 2x + 3 = 15?"   # scripted turns
    python -m app.cli --model qwen3:8b --no-trace

Shows the plan, each tool step with its intermediate result, verification,
and the final answer. Remembers earlier turns so follow-ups work.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Callable, TextIO

HELP = """Commands:
  /help            show this help
  /reset           forget the conversation
  /history         show what the agent remembers
  /trace on|off    show or hide the step-by-step work
  /exit            quit (also /quit, Ctrl+D)
Ask any math or word problem, then follow up: "what if ... instead?", "explain step 2"."""

BANNER = "Math agent. Ask a math or word problem. /help for commands, /exit to quit."


def _format_value(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def _format_call(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}({', '.join(f'{k}={_format_value(v)}' for k, v in arguments.items())})"


class Renderer:
    """Prints routing, plan and executor events as they arrive."""

    def __init__(self, out: TextIO | None = None, trace: bool = True):
        self._out = out  # None means "whatever sys.stdout is at write time"
        self.trace = trace
        self._message = ""

    def write(self, text: str = "") -> None:
        print(text, file=self._out or sys.stdout, flush=True)

    def start(self, message: str) -> None:
        self._message = message

    def __call__(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.trace:
            return
        if kind == "route":
            problem = payload.get("problem", "").strip()
            if payload.get("intent") == "solve" and problem and problem != self._message.strip():
                self.write(f"Interpreting as: {problem}")
                self.write()
        elif kind == "plan" and not payload.get("should_decline"):
            steps = payload.get("steps", [])
            self.write(f"Plan ({len(steps)} step{'s' if len(steps) != 1 else ''})")
            width = max((len(s["tool_name"]) for s in steps), default=0)
            for step in steps:
                self.write(f"  {step['step_number']}. {step['tool_name'].ljust(width)}  {step.get('purpose', '')}".rstrip())
            self.write()
        elif kind == "event":
            self._event(payload)

    def _event(self, event: dict[str, Any]) -> None:
        kind = event["event_type"]
        if kind == "tool_call_started":
            prefix = f"Step {event['step_number']}" + (f" (attempt {event['attempt']})" if event["attempt"] > 1 else "")
            self.write(f"{prefix}  {_format_call(event['tool_name'], event['arguments'])}")
        elif kind == "tool_call_succeeded":
            output = event["output"]
            self.write(f"        -> {output.get('result', output.get('exact'))}")
            for note in output.get("steps") or []:
                self.write(f"           - {note}")
        elif kind == "tool_call_failed":
            self.write(f"        ! {event['reason']}")
        elif kind == "step_repaired":
            reason = f"  ({event['reasoning']})" if event.get("reasoning") else ""
            self.write(f"        repairing: {_format_call(event['tool_name'], event['arguments'])}{reason}")

    def result(self, result) -> None:
        if result.kind == "solve":
            check = {True: " (verified)", False: " (verification FAILED)"}.get(result.verified, "")
            if self.trace:
                self.write()
            self.write(f"Answer: {result.reply}{check}")
        elif result.kind == "error":
            self.write(f"Sorry - {result.reply}")
        else:
            self.write(result.reply)


async def _read_line(prompt: str, stream: TextIO) -> str | None:
    if stream is sys.stdin and sys.stdin.isatty():
        try:
            return await asyncio.to_thread(input, prompt)
        except EOFError:
            return None
    line = await asyncio.to_thread(stream.readline)
    return None if line == "" else line.rstrip("\n")


async def handle_command(command: str, session, renderer: Renderer) -> bool:
    """Run a /command. Returns False when the chat should end."""
    name, _, arg = command.strip().partition(" ")
    name, arg = name.lower(), arg.strip().lower()
    if name in ("/exit", "/quit"):
        return False
    if name == "/help":
        renderer.write(HELP)
    elif name == "/reset":
        session.reset()
        renderer.write("Memory cleared.")
    elif name == "/history":
        renderer.write(session.memory.summary())
    elif name == "/trace":
        if arg not in ("on", "off"):
            renderer.write("Usage: /trace on|off")
        else:
            renderer.trace = arg == "on"
            renderer.write(f"Step-by-step trace {'on' if renderer.trace else 'off'}.")
    else:
        renderer.write(f"Unknown command {name}. Type /help.")
    return True


async def ask(session, message: str, renderer: Renderer) -> None:
    renderer.start(message)
    try:
        result = await session.ask(message, on_event=renderer)
    except Exception as exc:  # LLM/network problems should not kill the chat
        renderer.write(f"Sorry - something went wrong talking to the model: {type(exc).__name__}: {exc}")
        return
    renderer.result(result)


async def run_chat(session, renderer: Renderer, questions: list[str] | None = None, stream: TextIO = sys.stdin,
                   prompt: str = "you> ") -> None:
    """Scripted turns when ``questions`` is given, otherwise read lines until /exit or EOF."""
    if questions:
        for question in questions:
            renderer.write(f"{prompt}{question}")
            if question.startswith("/"):
                if not await handle_command(question, session, renderer):
                    return
            else:
                await ask(session, question, renderer)
            renderer.write()
        return

    renderer.write(BANNER)
    while True:
        line = await _read_line(prompt, stream)
        if line is None:
            renderer.write()
            return
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            if not await handle_command(line, session, renderer):
                return
            continue
        await ask(session, line, renderer)
        renderer.write()


def _configure_environment(args: argparse.Namespace) -> None:
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        key = "GEMINI_MODEL" if os.getenv("LLM_PROVIDER", "ollama").lower() == "gemini" else "OLLAMA_MODEL_2"
        os.environ[key] = args.model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Interactive math chat with step-by-step reasoning.")
    parser.add_argument("-q", "--question", action="append", help="Run these turns non-interactively (repeatable).")
    parser.add_argument("--provider", choices=["ollama", "gemini"], help="Override LLM_PROVIDER.")
    parser.add_argument("--model", help="Model to use (sets OLLAMA_MODEL_2 or GEMINI_MODEL).")
    parser.add_argument("--no-trace", action="store_true", help="Only print final answers.")
    parser.add_argument("--no-repair", action="store_true", help="Do not ask the LLM to repair failed steps.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, session_factory: Callable[..., Any] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)
    _configure_environment(args)
    if session_factory is None:
        from app.chat_session import build_session

        session_factory = build_session
    session = session_factory(repair=not args.no_repair)
    renderer = Renderer(trace=not args.no_trace)
    try:
        asyncio.run(run_chat(session, renderer, questions=args.question))
    except KeyboardInterrupt:
        renderer.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
