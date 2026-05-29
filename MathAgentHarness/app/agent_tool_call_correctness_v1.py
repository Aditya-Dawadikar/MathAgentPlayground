import os
from datetime import datetime

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


DEFAULT_SYSTEM_PROMPT = """Use the available tools when needed."""


@tool
def add(a: float, b: float) -> float:
    """Add two numeric values.

    Arguments must be named a and b.
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Arguments must be named a and b, where the operation is a - b.
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numeric values.

    Arguments must be named a and b.
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Arguments must be named a and b, where the operation is a / b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent.

    Arguments must be named base and exponent. Do not use a and b for this tool.
    """
    return base ** exponent


@tool
def get_current_datetime() -> str:
    """Return the current date and time.

    This tool takes no arguments.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [add, subtract, multiply, divide, power, get_current_datetime]


def build_agent(with_memory: bool = False):
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    checkpointer = MemorySaver() if with_memory else None
    return create_react_agent(
        llm,
        TOOLS,
        prompt=DEFAULT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
