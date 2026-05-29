import os
from datetime import datetime

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Use this for addition."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a. Use this for subtraction."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Use this for multiplication."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Use this for division."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent. Use this for exponentiation."""
    return base ** exponent


@tool
def get_current_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [add, subtract, multiply, divide, power, get_current_datetime]


def build_agent(with_memory: bool = False):
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    checkpointer = MemorySaver() if with_memory else None
    return create_react_agent(llm, TOOLS, checkpointer=checkpointer)
