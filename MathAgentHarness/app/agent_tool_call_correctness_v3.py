import os
from datetime import datetime

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


DEFAULT_SYSTEM_PROMPT = """You are a tool-using assistant.

You have access to these tools and should use them explicitly when the user asks for the matching operation:
- add(a, b): use for addition, plus, total, or sum requests.
- subtract(a, b): use for subtraction, minus, difference, or take-away requests. The operation is a - b.
- multiply(a, b): use for multiplication, times, or product requests.
- divide(a, b): use for division or quotient requests. The operation is a / b.
- power(base, exponent): use for exponentiation requests such as raised to the power, squared, or cubed.
- get_current_datetime(): use when the user asks for the current date, current time, or current date and time.

Tool-calling rules:
- For arithmetic or date/time requests, call the relevant tool instead of answering directly.
- Match the tool argument names exactly.
- For add, subtract, multiply, and divide, provide arguments named a and b.
- For power, provide arguments named base and exponent.
- Do not invent extra arguments.
- If a tool applies, make the tool call before giving a final answer.
"""


@tool
def add(a: float, b: float) -> float:
    """Add two numeric values.

    Use this for addition requests such as plus, add, total, or sum.
    Format the tool input as {"a": <first number>, "b": <second number>}.
    Arguments must be named a and b.
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Use this for subtraction requests such as subtract, minus, difference, or take away.
    Format the tool input as {"a": <starting number>, "b": <number to subtract>}.
    Arguments must be named a and b, where the operation is a - b.
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numeric values.

    Use this for multiplication requests such as multiply, times, or product.
    Format the tool input as {"a": <first factor>, "b": <second factor>}.
    Arguments must be named a and b.
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Use this for division requests such as divide or quotient.
    Format the tool input as {"a": <dividend>, "b": <divisor>}.
    Arguments must be named a and b, where the operation is a / b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent.

    Use this for exponentiation requests such as power, raised to, squared, or cubed.
    Format the tool input as {"base": <base number>, "exponent": <power>}.
    Arguments must be named base and exponent. Do not use a and b for this tool.
    """
    return base ** exponent


@tool
def get_current_datetime() -> str:
    """Return the current date and time.

    Use this when the user asks for the current date, current time, or both.
    Format the tool input as {} because this tool takes no arguments.
    This tool takes no arguments.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [add, subtract, multiply, divide, power, get_current_datetime]


def build_agent(with_memory: bool = False):
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    checkpointer = MemorySaver() if with_memory else None
    return create_react_agent(
        llm,
        TOOLS,
        prompt=DEFAULT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )