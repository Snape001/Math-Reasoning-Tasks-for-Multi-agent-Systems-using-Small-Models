from __future__ import annotations

import ast
import math
import re


def _arithmetic(expression: str) -> float | None:
    """Evaluate a numeric Asymptote coordinate without executing code."""

    try:
        node = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return None

    def visit(item: ast.AST) -> float:
        if isinstance(item, ast.Expression):
            return visit(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.UnaryOp) and isinstance(
            item.op, (ast.UAdd, ast.USub)
        ):
            value = visit(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp) and isinstance(
            item.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = visit(item.left), visit(item.right)
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Sub):
                return left - right
            if isinstance(item.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("unsupported coordinate expression")

    try:
        return visit(node)
    except (ValueError, ZeroDivisionError):
        return None


def asy_label_grounding(problem: str) -> str:
    """Map coordinate-positioned Asymptote labels to their nearest named point.

    This uses coordinates only to locate annotations, never to infer lengths or
    angles.  It is intentionally domain-general and contains no problem answers.
    """

    blocks = re.findall(r"\[asy\](.*?)\[/asy\]", problem, flags=re.DOTALL)
    if not blocks:
        return "NOT_APPLICABLE"
    code = "\n".join(blocks)

    points: dict[str, tuple[float, float]] = {}
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*"
        r"\(\s*([^,;]+?)\s*,\s*([^)]+?)\s*\)\s*;?",
        code,
    ):
        x = _arithmetic(match.group(2))
        y = _arithmetic(match.group(3))
        if x is not None and y is not None:
            points[match.group(1)] = (x, y)

    if not points:
        return "NO_PARSEABLE_POINTS"

    pair_declarations = re.findall(r"\bpair\s+([^;]+);", code)
    anchor_names: list[str] = []
    if pair_declarations:
        anchor_names = re.findall(
            r"\b[A-Za-z][A-Za-z0-9_]*\b", pair_declarations[0]
        )
    anchors = {
        name: points[name] for name in anchor_names if name in points
    } or points

    grounded: list[str] = []
    pattern = re.compile(
        r'label\(\s*"([^"]+)"\s*,\s*'
        r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
    )
    for match in pattern.finditer(code):
        x = _arithmetic(match.group(2))
        y = _arithmetic(match.group(3))
        if x is None or y is None:
            continue
        distances = sorted(
            (
                math.hypot(x - px, y - py),
                name,
            )
            for name, (px, py) in anchors.items()
        )
        nearest_distance, nearest_name = distances[0]
        label = match.group(1).replace("$", "")
        grounded.append(
            f"label {label!r} at ({x:g},{y:g}) is nearest to point "
            f"{nearest_name} (distance {nearest_distance:.2f})"
        )

    if not grounded:
        return "NO_COORDINATE_POSITIONED_LABELS"
    return "\n".join(grounded)
