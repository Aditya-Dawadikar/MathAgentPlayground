"""Interactive math chat in the terminal (issue #13).

    python -m app.cli                      # interactive chat
    python -m app.cli -q "Solve 2x + 3 = 11" -q "what if it were 2x + 3 = 15?"   # scripted turns
    python -m app.cli --model qwen3:8b --no-trace
    python -m app.cli --no-color          # plain text (also NO_COLOR=1; FORCE_COLOR=1 forces color)

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

# ANSI SGR codes per semantic role. Change the look of the CLI here.
STYLES = {
    "banner": "1;36",      # bold cyan
    "prompt": "1;36",      # bold cyan "you>"
    "interpret": "35",     # magenta "Interpreting as:"
    "heading": "1;34",     # bold blue "Plan (n steps)"
    "step": "1",           # bold "Step 1"
    "tool": "36",          # cyan tool names
    "muted": "2",          # dim purposes, notes, numbering
    "result": "32",        # green intermediate results
    "error": "31",         # red failures
    "warn": "33",          # yellow repairs, notices
    "answer": "1;32",      # bold green "Answer: ..."
    "failed": "1;31",      # bold red "(verification FAILED)"
}


def _enable_windows_vt(stream: TextIO) -> bool:
    """Turn on ANSI escape handling in classic Windows consoles; True if it worked."""
    try:
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        return False


def supports_color(stream: TextIO) -> bool:
    """Color for interactive terminals only, honouring NO_COLOR / FORCE_COLOR."""
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    if os.getenv("TERM") == "dumb" or not hasattr(stream, "isatty") or not stream.isatty():
        return False
    return _enable_windows_vt(stream) if os.name == "nt" else True


def _format_value(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


class Renderer:
    """Prints routing, plan and executor events as they arrive."""

    def __init__(self, out: TextIO | None = None, trace: bool = True, color: bool | None = None):
        """``color``: True/False, or None to detect (terminal yes, files and pipes no)."""
        self._out = out  # None means "whatever sys.stdout is at write time"
        self.trace = trace
        self.color = supports_color(out or sys.stdout) if color is None else color
        self._message = ""

    def write(self, text: str = "") -> None:
        print(text, file=self._out or sys.stdout, flush=True)

    def paint(self, text: Any, style: str) -> str:
        """Wrap text in the ANSI code for a style from STYLES (no-op without color)."""
        text = str(text)
        if not self.color or not text:
            return text
        return f"\033[{STYLES[style]}m{text}\033[0m"

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        args = ", ".join(f"{k}={_format_value(v)}" for k, v in arguments.items())
        return f"{self.paint(tool_name, 'tool')}({args})"

    def prompt(self, text: str) -> str:
        return self.paint(text, "prompt")

    def start(self, message: str) -> None:
        self._message = message

    def __call__(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.trace:
            return
        if kind == "route":
            problem = payload.get("problem", "").strip()
            if payload.get("intent") == "solve" and problem and problem != self._message.strip():
                self.write(f"{self.paint('Interpreting as:', 'interpret')} {problem}")
                self.write()
        elif kind == "plan" and not payload.get("should_decline"):
            steps = payload.get("steps", [])
            self.write(self.paint(f"Plan ({len(steps)} step{'s' if len(steps) != 1 else ''})", "heading"))
            width = max((len(s["tool_name"]) for s in steps), default=0)
            for step in steps:
                number = self.paint(f"{step['step_number']}.", "muted")
                purpose = self.paint(step.get("purpose", ""), "muted")
                self.write(f"  {number} {self.paint(step['tool_name'].ljust(width), 'tool')}  {purpose}".rstrip())
            self.write()
        elif kind == "event":
            self._event(payload)

    def _event(self, event: dict[str, Any]) -> None:
        kind = event["event_type"]
        if kind == "tool_call_started":
            prefix = f"Step {event['step_number']}" + (f" (attempt {event['attempt']})" if event["attempt"] > 1 else "")
            self.write(f"{self.paint(prefix, 'step')}  {self.call(event['tool_name'], event['arguments'])}")
        elif kind == "tool_call_succeeded":
            output = event["output"]
            self.write(f"        -> {self.paint(output.get('result', output.get('exact')), 'result')}")
            for note in output.get("steps") or []:
                self.write(self.paint(f"           - {note}", "muted"))
        elif kind == "tool_call_failed":
            self.write(self.paint(f"        ! {event['reason']}", "error"))
        elif kind == "step_repaired":
            reason = self.paint(f"  ({event['reasoning']})", "muted") if event.get("reasoning") else ""
            self.write(f"        {self.paint('repairing:', 'warn')} {self.call(event['tool_name'], event['arguments'])}{reason}")

    def result(self, result) -> None:
        if result.kind == "solve":
            check = {True: self.paint(" (verified)", "result"),
                     False: self.paint(" (verification FAILED)", "failed")}.get(result.verified, "")
            if self.trace:
                self.write()
            self.write(f"{self.paint('Answer:', 'answer')} {self.paint(result.reply, 'answer')}{check}")
        elif result.kind == "error":
            self.write(self.paint(f"Sorry - {result.reply}", "error"))
        elif result.kind in ("clarify", "decline"):
            self.write(self.paint(result.reply, "warn"))
        else:
            self.write(result.reply)

    def history(self, summary: str) -> None:
        for line in summary.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("Turn ", "Recent turns", "Older turns")):
                line = self.paint(line, "step")
            elif stripped.startswith("final answer:"):
                line = self.paint(line, "result")
            elif stripped.startswith(("interpreted as:", "[")) or "(no earlier turns)" in stripped:
                line = self.paint(line, "muted")
            self.write(line)


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
        renderer.write(renderer.paint("Memory cleared.", "warn"))
    elif name == "/history":
        renderer.history(session.memory.summary())
    elif name == "/trace":
        if arg not in ("on", "off"):
            renderer.write(renderer.paint("Usage: /trace on|off", "warn"))
        else:
            renderer.trace = arg == "on"
            renderer.write(renderer.paint(f"Step-by-step trace {'on' if renderer.trace else 'off'}.", "warn"))
    else:
        renderer.write(renderer.paint(f"Unknown command {name}. Type /help.", "error"))
    return True


async def ask(session, message: str, renderer: Renderer) -> None:
    renderer.start(message)
    try:
        result = await session.ask(message, on_event=renderer)
    except Exception as exc:  # LLM/network problems should not kill the chat
        renderer.write(renderer.paint(f"Sorry - something went wrong talking to the model: {type(exc).__name__}: {exc}", "error"))
        return
    renderer.result(result)


async def run_chat(session, renderer: Renderer, questions: list[str] | None = None, stream: TextIO = sys.stdin,
                   prompt: str = "you> ") -> None:
    """Scripted turns when ``questions`` is given, otherwise read lines until /exit or EOF."""
    if questions:
        for question in questions:
            renderer.write(f"{renderer.prompt(prompt)}{question}")
            if question.startswith("/"):
                if not await handle_command(question, session, renderer):
                    return
            else:
                await ask(session, question, renderer)
            renderer.write()
        return

    renderer.write(renderer.paint(BANNER, "banner"))
    while True:
        line = await _read_line(renderer.prompt(prompt), stream)
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
    parser.add_argument("--no-color", action="store_true", help="Plain text output (also NO_COLOR=1).")
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
    renderer = Renderer(trace=not args.no_trace, color=False if args.no_color else None)
    try:
        asyncio.run(run_chat(session, renderer, questions=args.question))
    except KeyboardInterrupt:
        renderer.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
