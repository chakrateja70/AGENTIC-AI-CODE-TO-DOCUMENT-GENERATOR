from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import tool

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "factorial": math.factorial,
    "fabs": math.fabs,
    "pow": math.pow,
    "abs": abs,
    "round": round,
}
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_eval_node(arg) for arg in node.args]
        return _FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression from a restricted AST grammar (no eval/exec)."""
    parsed = ast.parse(expression, mode="eval")
    return _eval_node(parsed.body)


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '2 + 2 * (3 - 1)', 'sqrt(16)', 'sin(pi / 2)'.
    Supported functions: sqrt, sin, cos, tan, log, log10, exp, factorial, fabs, pow, abs, round.
    """
    try:
        result = safe_eval(expression)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"
    return str(result)


TOOLS = [calculator]
