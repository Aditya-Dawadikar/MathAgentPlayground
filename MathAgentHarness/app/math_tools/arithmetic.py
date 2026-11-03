"""Core arithmetic tools with exact results (issue #2)."""

from __future__ import annotations

import sympy as sp

from .core import MathToolError, Number, ensure_defined, math_tool, ok, parse_expression, to_integer, to_number


def _binary(a: Number, b: Number) -> tuple[sp.Expr, sp.Expr]:
    return to_number(a, "a"), to_number(b, "b")


@math_tool
def add(a: Number, b: Number) -> dict:
    """Add two numbers: a + b.

    Use for plus, sum, total, or "more than". Arguments are numbers or exact
    numeric strings like "1/3". Example: {"a": 2, "b": 3} -> 5.
    """
    x, y = _binary(a, b)
    return ok(x + y)


@math_tool
def subtract(a: Number, b: Number) -> dict:
    """Subtract b from a: a - b.

    Use for minus, difference, "less than", or take away. Order matters.
    Example: {"a": 10, "b": 4} -> 6.
    """
    x, y = _binary(a, b)
    return ok(x - y)


@math_tool
def multiply(a: Number, b: Number) -> dict:
    """Multiply two numbers: a * b.

    Use for times, product, "of", or doubling/tripling. Example: {"a": 6, "b": 7} -> 42.
    """
    x, y = _binary(a, b)
    return ok(x * y)


@math_tool
def divide(a: Number, b: Number) -> dict:
    """Divide a by b: a / b, returning an exact fraction plus a decimal.

    Use for division, quotient, per, ratio, or sharing equally. Order matters.
    Example: {"a": 1, "b": 3} -> exact "1/3", decimal 0.333...
    """
    x, y = _binary(a, b)
    if y == 0:
        raise MathToolError("math_error", "division by zero is undefined")
    return ok(x / y)


@math_tool
def power(base: Number, exponent: Number) -> dict:
    """Raise base to an exponent: base ** exponent.

    Use for powers, squared (exponent 2), cubed (exponent 3), or negative/fractional
    exponents. Example: {"base": 2, "exponent": 10} -> 1024.
    """
    b, e = to_number(base, "base"), to_number(exponent, "exponent")
    if b == 0 and e.is_negative:
        raise MathToolError("math_error", "0 raised to a negative power is undefined")
    return ok(ensure_defined(b**e))


def _root_result(value: sp.Expr, x: sp.Expr, n: int) -> dict:
    value = sp.nsimplify(value) if value.is_Float else value
    is_real = bool(value.is_real)
    steps = []
    if not is_real:
        steps.append(f"{x} is negative and the root index {n} is even, so the root is not a real number")
    return ok(value, steps=steps, is_real=is_real)


@math_tool
def sqrt(x: Number) -> dict:
    """Square root of x in simplest exact radical form.

    Example: {"x": 72} -> "6*sqrt(2)" (decimal 8.485...). Negative inputs return an
    imaginary result with is_real=false.
    """
    value = to_number(x, "x")
    return _root_result(sp.sqrt(value), value, 2)


@math_tool
def nth_root(x: Number, n: int) -> dict:
    """The n-th root of x, e.g. cube root when n=3.

    Odd roots of negative numbers are real (nth_root(-8, 3) = -2). Even roots of
    negative numbers return an imaginary result with is_real=false.
    Example: {"x": 27, "n": 3} -> 3.
    """
    value = to_number(x, "x")
    index = int(to_integer(n, "n"))
    if index < 1:
        raise MathToolError("invalid_input", "n must be a positive integer")
    if value.is_negative and index % 2 == 1:
        root = sp.real_root(value, index)
    else:
        root = sp.root(value, index)
    return _root_result(root, value, index)


@math_tool
def modulo(a: Number, b: Number) -> dict:
    """Remainder when a is divided by b (a mod b).

    Example: {"a": 17, "b": 5} -> 2, with quotient 3.
    """
    x, y = _binary(a, b)
    if y == 0:
        raise MathToolError("math_error", "modulo by zero is undefined")
    return ok(sp.Mod(x, y), quotient=sp.floor(x / y))


@math_tool
def absolute_value(x: Number) -> dict:
    """Absolute value |x| (distance from zero). Example: {"x": -7.5} -> 15/2 (7.5)."""
    return ok(sp.Abs(to_number(x, "x")))


@math_tool
def gcd(a: Number, b: Number) -> dict:
    """Greatest common divisor (greatest common factor) of two integers.

    Example: {"a": 48, "b": 18} -> 6.
    """
    return ok(sp.gcd(to_integer(a, "a"), to_integer(b, "b")))


@math_tool
def lcm(a: Number, b: Number) -> dict:
    """Least common multiple of two integers (useful for common denominators).

    Example: {"a": 4, "b": 6} -> 12.
    """
    return ok(sp.lcm(to_integer(a, "a"), to_integer(b, "b")))


@math_tool
def prime_factorize(n: Number) -> dict:
    """Prime factorization of an integer greater than 1.

    Example: {"n": 360} -> "2^3 * 3^2 * 5", factors {"2": 3, "3": 2, "5": 1}.
    """
    value = to_integer(n, "n")
    if value < 2:
        raise MathToolError("invalid_input", "n must be an integer greater than 1")
    factors = sp.factorint(value)
    text = " * ".join(f"{p}^{k}" if k > 1 else str(p) for p, k in sorted(factors.items()))
    return ok(value, result=text, factors=factors, is_prime=sp.isprime(value))


@math_tool
def fraction_simplify(numerator: Number, denominator: Number) -> dict:
    """Reduce a fraction to lowest terms.

    Example: {"numerator": 42, "denominator": 56} -> "3/4" (divided by gcd 14).
    """
    top, bottom = to_number(numerator, "numerator"), to_number(denominator, "denominator")
    if bottom == 0:
        raise MathToolError("math_error", "denominator cannot be zero")
    steps = []
    if top.is_integer and bottom.is_integer:
        divisor = sp.gcd(top, bottom)
        steps.append(f"divide numerator and denominator by gcd {divisor}")
    value = top / bottom
    numer, denom = sp.fraction(value)
    return ok(value, steps=steps, numerator=numer, denominator=denom)


@math_tool
def evaluate_expression(expression: str) -> dict:
    """Evaluate a numeric expression exactly, respecting order of operations.

    Supports + - * / ^, parentheses, sqrt, abs, log, sin/cos/tan (radians), pi, e.
    The expression must not contain variables (use substitute for that).
    Example: {"expression": "(18+7)*4 - 2^3"} -> 92.
    """
    expr = parse_expression(expression)
    if expr.free_symbols:
        names = ", ".join(sorted(s.name for s in expr.free_symbols))
        raise MathToolError("invalid_input", f"expression contains variables ({names}); use substitute or simplify instead")
    value = ensure_defined(sp.simplify(expr), "expression")
    return ok(value)


TOOLS = [
    add,
    subtract,
    multiply,
    divide,
    power,
    sqrt,
    nth_root,
    modulo,
    absolute_value,
    gcd,
    lcm,
    prime_factorize,
    fraction_simplify,
    evaluate_expression,
]
