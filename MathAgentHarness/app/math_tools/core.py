"""Shared conventions for every math tool.

Parsing
    All user/LLM supplied math goes through :func:`parse_expression`,
    :func:`parse_relation` or :func:`to_number`. Parsing is sandboxed: the
    evaluation namespace only contains SymPy number/symbol constructors plus an
    explicit allow-list of functions and constants, Python builtins are
    unreachable, and characters such as quotes, brackets and ``__`` are
    rejected up front. Implicit multiplication (``2x``, ``3(x-1)``) and ``^``
    for powers are supported.

Result envelope
    Every tool returns a JSON-serialisable dict::

        {"result": "4", "exact": "4", "decimal": 4.0, "latex": "4", "steps": [...], ...extra}

    ``exact`` is always a SymPy-parseable string, so it can be fed straight
    back into another tool. ``decimal`` is a convenience float (``None`` when
    the value is symbolic or non-real).

Error envelope
    Tools never raise. Failures come back as::

        {"error": "<code>", "message": "<human readable explanation>"}

    Codes: ``parse_error``, ``invalid_input``, ``math_error``, ``no_solution``,
    ``unsupported``, ``timeout``, ``internal_error``.

Timeouts
    Each tool call runs in a daemon thread and is abandoned after
    ``MATH_TOOL_TIMEOUT_SECONDS`` (default 10s) so a pathological
    ``solve``/``simplify`` cannot hang the agent.
"""

from __future__ import annotations

import functools
import os
import re
import string
import threading
from typing import Any, Callable, Union

import sympy as sp
from langchain_core.tools import tool as lc_tool
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Numbers may be passed as JSON numbers or as exact strings such as "1/3" or "sqrt(2)".
Number = Union[int, float, str]

DEFAULT_TIMEOUT_SECONDS = float(os.getenv("MATH_TOOL_TIMEOUT_SECONDS", "10"))
MAX_INPUT_LENGTH = 500

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Only what SymPy's generated parse code needs. An empty __builtins__ blocks Python builtins.
_GLOBALS: dict[str, Any] = {
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "Function": sp.Function,
    "Number": sp.Number,
    "__builtins__": {},
}


def _log2(x):
    return sp.log(x, 2)


def _log10(x):
    return sp.log(x, 10)


_FUNCTIONS: dict[str, Any] = {
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "root": sp.root,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log2": _log2,
    "log10": _log10,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "factorial": sp.factorial,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "I": sp.I,
    "oo": sp.oo,
    "inf": sp.oo,
    "infinity": sp.oo,
}

# Unicode math that shows up in pasted word problems, mapped to parser syntax.
_UNICODE_MATH = {
    "\u00b2": "^2", "\u00b3": "^3", "\u00b9": "^1", "\u2070": "^0", "\u2074": "^4", "\u2075": "^5",
    "\u00d7": "*", "\u22c5": "*", "\u00b7": "*", "\u2217": "*", "\u00f7": "/", "\u2044": "/", "\u2215": "/",
    "\u2212": "-", "\u2013": "-", "\u2014": "-",
    "\u2264": "<=", "\u2265": ">=", "\u2260": "!=", "\u03c0": "pi", "\u221e": "oo",
    "\u00bd": "(1/2)", "\u2153": "(1/3)", "\u2154": "(2/3)", "\u00bc": "(1/4)", "\u00be": "(3/4)",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",
}
_SQRT_GROUP = re.compile(r"\u221a\s*\(")
_SQRT_ATOM = re.compile(r"\u221a\s*([A-Za-z0-9.]+)")


def normalize_math_text(text: str) -> str:
    """Rewrite unicode math (x\u00b2, \u00d7, \u00f7, \u2212, \u2264, \u03c0, \u221a, \u00bd) as parser syntax."""
    text = _SQRT_GROUP.sub("sqrt(", text)
    text = _SQRT_ATOM.sub(r"sqrt(\1)", text)
    for symbol, replacement in _UNICODE_MATH.items():
        text = text.replace(symbol, replacement)
    return text


_FORBIDDEN_CHARS = re.compile(r"[^A-Za-z0-9_\s+\-*/^().,=<>!]")
_RELATION_SPLIT = re.compile(r"(<=|>=|!=|==|=|<|>)")
_VARIABLE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class MathToolError(Exception):
    """Raised inside tools; converted into the error envelope by :func:`math_tool`."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- parsing


def _local_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {c: sp.Symbol(c) for c in string.ascii_letters}
    namespace.update(_FUNCTIONS)
    return namespace


def _check_text(text: str, label: str) -> str:
    if not isinstance(text, str):
        raise MathToolError("invalid_input", f"{label} must be a string, got {type(text).__name__}")
    text = normalize_math_text(text).strip()
    if not text:
        raise MathToolError("parse_error", f"{label} is empty")
    if len(text) > MAX_INPUT_LENGTH:
        raise MathToolError("invalid_input", f"{label} is longer than {MAX_INPUT_LENGTH} characters")
    bad = _FORBIDDEN_CHARS.search(text)
    if bad:
        raise MathToolError("parse_error", f"unsupported character {bad.group()!r} in {label} {text!r}")
    if "__" in text or re.search(r"\.\s*[A-Za-z_]", text):
        raise MathToolError("parse_error", f"unsupported syntax in {label} {text!r}")
    return text


def _parse(text: str, label: str) -> sp.Expr:
    try:
        expr = parse_expr(
            text,
            local_dict=_local_namespace(),
            global_dict=dict(_GLOBALS),
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except MathToolError:
        raise
    except Exception as exc:  # SyntaxError, TokenError, TypeError, ...
        raise MathToolError("parse_error", f"could not parse {label} {text!r}: {exc}") from exc
    if not isinstance(expr, sp.Basic):
        raise MathToolError("parse_error", f"{label} {text!r} is not a math expression")
    return expr


def parse_expression(value: Number, label: str = "expression") -> sp.Expr:
    """Parse an expression (no ``=``/``<``/``>``). Numbers pass through exactly."""
    if isinstance(value, bool):
        raise MathToolError("invalid_input", f"{label} must be a number or expression, got a boolean")
    if isinstance(value, (int, float)):
        return to_number(value, label)
    text = _check_text(value, label)
    if _RELATION_SPLIT.search(text):
        raise MathToolError("invalid_input", f"{label} should be an expression, not an equation or inequality: {text!r}")
    return _parse(text, label)


def parse_relation(text: str, label: str = "equation", allow_bare_expression: bool = True):
    """Parse ``lhs = rhs`` or an (optionally chained) inequality such as ``1 < 2x + 3 <= 7``.

    A bare expression is treated as ``expression = 0`` when ``allow_bare_expression``.
    Returns a SymPy ``Eq``/relational, or ``And`` for chained inequalities.
    """
    text = _check_text(text, label)
    parts = _RELATION_SPLIT.split(text)
    sides, ops = parts[0::2], parts[1::2]
    if not ops:
        if not allow_bare_expression:
            raise MathToolError("invalid_input", f"{label} needs '=' or an inequality sign: {text!r}")
        return sp.Eq(_parse(text, label), 0, evaluate=False)
    if any(not side.strip() for side in sides):
        raise MathToolError("parse_error", f"{label} {text!r} has an empty side")
    exprs = [_parse(side, label) for side in sides]
    relations = []
    for left, op, right in zip(exprs, ops, exprs[1:]):
        if op in ("=", "=="):
            relations.append(sp.Eq(left, right, evaluate=False))
        elif op == "!=":
            relations.append(sp.Ne(left, right, evaluate=False))
        else:
            relations.append(sp.Rel(left, right, op, evaluate=False))
    return relations[0] if len(relations) == 1 else sp.And(*relations, evaluate=False)


def parse_equation(text: str, label: str = "equation") -> sp.Eq:
    """Parse exactly one equation ``lhs = rhs`` (bare expressions mean ``= 0``)."""
    relation = parse_relation(text, label)
    if not isinstance(relation, sp.Equality):
        raise MathToolError("invalid_input", f"{label} must be an equation with a single '=': {text!r}")
    return relation


def to_number(value: Number, label: str = "value") -> sp.Expr:
    """Convert a JSON number or numeric string to an exact SymPy number.

    Floats are converted through their decimal string, so ``0.05`` becomes ``1/20``.
    """
    if isinstance(value, bool):
        raise MathToolError("invalid_input", f"{label} must be a number, got a boolean")
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise MathToolError("invalid_input", f"{label} must be a finite number")
        return sp.Integer(int(value)) if value.is_integer() else sp.Rational(repr(value))
    expr = parse_expression(value, label)
    if expr.free_symbols:
        raise MathToolError("invalid_input", f"{label} must be numeric, got {value!r}")
    return expr


def to_integer(value: Number, label: str = "value") -> sp.Integer:
    number = to_number(value, label)
    if not number.is_integer:
        raise MathToolError("invalid_input", f"{label} must be an integer, got {value!r}")
    return sp.Integer(number)


def parse_variable(name: str | None, label: str = "variable") -> sp.Symbol:
    if not isinstance(name, str) or not _VARIABLE_NAME.match(name.strip()):
        raise MathToolError("invalid_input", f"{label} must be a variable name like 'x', got {name!r}")
    name = name.strip()
    if name in _FUNCTIONS:
        raise MathToolError("invalid_input", f"{label} {name!r} is a reserved name")
    return sp.Symbol(name)


def resolve_variable(expr: sp.Basic, variable: str | None = None) -> sp.Symbol:
    """Use the given variable, or infer it when the expression has exactly one symbol."""
    if variable:
        return parse_variable(variable)
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(symbols) == 1:
        return symbols[0]
    if not symbols:
        raise MathToolError("invalid_input", "the input has no variable")
    names = ", ".join(s.name for s in symbols)
    raise MathToolError("invalid_input", f"several variables ({names}); specify which one to use")


def ensure_defined(value: sp.Basic, context: str = "result") -> sp.Basic:
    """Reject undefined results such as ``1/0`` (zoo) or ``0/0`` (nan)."""
    if isinstance(value, sp.Basic) and value.has(sp.zoo, sp.nan):
        raise MathToolError("math_error", f"{context} is undefined (division by zero or indeterminate form)")
    return value


# --------------------------------------------------------------------------- envelopes


def to_jsonable(value: Any) -> Any:
    """Recursively convert SymPy objects into JSON-friendly values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if value is sp.true or value is sp.false:
        return bool(value)
    if isinstance(value, sp.Integer):
        return int(value)
    if isinstance(value, sp.Basic):
        return str(value)
    if isinstance(value, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items = sorted(items, key=sp.default_sort_key)
        return [to_jsonable(v) for v in items]
    return str(value)


def decimal_value(value: Any) -> float | None:
    """Float approximation of a real numeric value, otherwise ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, sp.Basic) or value.free_symbols or not getattr(value, "is_number", False):
        return None
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    approx = sp.N(value, 15)
    if approx.is_real:
        return float(approx)
    return None


def ok(value: Any = None, *, result: str | None = None, steps: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    """Build the success envelope."""
    envelope: dict[str, Any] = {}
    if value is sp.true or value is sp.false:
        value = bool(value)
    if isinstance(value, sp.Basic):
        envelope["result"] = result if result is not None else str(value)
        envelope["exact"] = str(value)
        envelope["decimal"] = decimal_value(value)
        envelope["latex"] = sp.latex(value)
    else:
        jsonable = to_jsonable(value)
        envelope["result"] = result if result is not None else (str(jsonable) if jsonable is not None else "")
        envelope["exact"] = jsonable
        envelope["decimal"] = decimal_value(value)
        envelope["latex"] = None
    envelope["steps"] = list(steps or [])
    for key, item in extra.items():
        envelope[key] = to_jsonable(item)
    return envelope


def err(code: str, message: str) -> dict[str, str]:
    """Build the error envelope."""
    return {"error": code, "message": message}


def _error_from_exception(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, MathToolError):
        return err(exc.code, exc.message)
    if isinstance(exc, ZeroDivisionError):
        return err("math_error", f"division by zero: {exc}")
    if isinstance(exc, NotImplementedError):
        return err("unsupported", f"this problem type is not supported: {exc}")
    if isinstance(exc, (ValueError, TypeError)):
        return err("invalid_input", f"{type(exc).__name__}: {exc}")
    return err("internal_error", f"{type(exc).__name__}: {exc}")


def call_guarded(func: Callable[..., Any], args: tuple, kwargs: dict, timeout: float) -> Any:
    """Run ``func`` with a timeout, converting every failure into an error envelope."""
    outcome: dict[str, Any] = {}

    def target():
        try:
            outcome["value"] = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - tools must never raise
            outcome["exc"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return err("timeout", f"{func.__name__} did not finish within {timeout:g} seconds")
    if "exc" in outcome:
        return _error_from_exception(outcome["exc"])
    return outcome.get("value")


def math_tool(func: Callable[..., Any] | None = None, *, name: str | None = None, timeout: float | None = None):
    """Decorator turning a plain function into a guarded LangChain tool.

    The wrapped function is also exposed as ``<tool>.func`` (LangChain's
    attribute), so it can be called directly in tests and other tools.
    """

    def decorate(f: Callable[..., Any]):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return call_guarded(f, args, kwargs, timeout or DEFAULT_TIMEOUT_SECONDS)

        return lc_tool(name or f.__name__)(wrapper)

    return decorate(func) if func is not None else decorate
