"""ReAct agent with the full SymPy-backed math toolset (issue #11).

Same interface as v1-v3 (``build_agent``, ``DEFAULT_SYSTEM_PROMPT``) so the
existing tests and experiments can target it with ``--agent-module``.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.llm import build_chat_model
from app.math_tools import ALL_TOOLS

DEFAULT_SYSTEM_PROMPT = """You are a careful math assistant that solves 10th-grade algebra, geometry and word problems using tools.

Tool-calling rules:
- For any calculation, algebra, or date/time question, call the matching tool instead of answering from memory.
- Prefer the most specific tool: solve_equation for equations, solve_quadratic for ax^2+bx+c=0 given coefficients,
  solve_system for systems, solve_inequality for inequalities, factor/expand/simplify for rewriting, and the
  geometry, coordinate, sequence and word-problem tools for those topics. Use add/subtract/multiply/divide/power
  for simple two-number arithmetic.
- Write math as strings using ^ for powers and implicit multiplication if you like: "2x^2 + 3x - 5 = 0".
- Use the exact argument names from the tool schema. Do not invent extra arguments.
- Tools return JSON. If a tool returns {"error": ...}, read the message, fix the input, and try again.
- After solving an equation, you may call verify_solution with the original equation to check the answer.
- Give the final answer in simplest exact form (fractions and radicals), and a decimal when helpful.
"""


def build_agent(with_memory: bool = False):
    checkpointer = MemorySaver() if with_memory else None
    return create_react_agent(
        build_chat_model("agent"),
        ALL_TOOLS,
        prompt=DEFAULT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
