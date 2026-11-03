"""Non-math helper tools kept alongside the math toolset."""

from __future__ import annotations

from datetime import datetime

from .core import math_tool, ok


@math_tool
def get_current_datetime() -> dict:
    """Return the current local date and time (YYYY-MM-DD HH:MM:SS). Takes no arguments."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ok(now, result=now)


TOOLS = [get_current_datetime]
