from __future__ import annotations

import ast
import operator
from typing import Any, Dict

from .base import Agent, AgentInput, AgentResult

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return float(node.n)  # type: ignore[return-value]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("unsupported_expression")


class CalculatorAgent(Agent):
    name = "calculator"

    async def run(self, payload: AgentInput) -> AgentResult:
        expr = payload.context.get("expression") or payload.message
        try:
            tree = ast.parse(expr, mode="eval")
            result = _eval_expr(tree.body)  # type: ignore[arg-type]
            return AgentResult(status="ok", data={"result": result})
        except Exception as exc:
            return AgentResult(status="error", error={"message": str(exc)})
