import ast
import re
from typing import Any


ANSWER_FIELDS = (
    "FINAL_ANSWER",
    "PROPOSED_ANSWER",
    "VERIFIED_ANSWER",
    "CHECKED_ANSWER",
    "CORRECTED_ANSWER",
    "ANSWER",
)

def remove_text_units(answer: str) -> str:
    """
    删除LaTeX文字单位及其平方/立方标记。

    例如：
    864 \\mbox{ inches}^2 -> 864
    """

    return re.sub(
        r"\\(?:mbox|text|mathrm)\s*"
        r"\{[^{}]*\}"
        r"(?:\^\{?[23]\}?)?",
        "",
        answer,
    )


def extract_boxed_answer(text: str) -> str | None:
    """
    提取最后一个完整的 \\boxed{...}，
    支持 \\boxed{\\frac{1}{2}} 这种嵌套大括号。
    """

    marker = r"\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    index = start + len(marker)
    depth = 1
    answer_chars: list[str] = []

    while index < len(text):
        char = text[index]

        if char == "{":
            depth += 1
            answer_chars.append(char)

        elif char == "}":
            depth -= 1

            if depth == 0:
                return "".join(answer_chars).strip()

            answer_chars.append(char)

        else:
            answer_chars.append(char)

        index += 1

    return None


def extract_declared_answer(text: str | None) -> str | None:
    """提取 answer-first 协议中的答案，并兼容旧的 ``\boxed{}`` 输出。"""

    if not text:
        return None

    boxed = extract_boxed_answer(text)
    field_text = text.replace("**", "")

    for field in ANSWER_FIELDS:
        flexible_field = re.escape(field).replace("_", r"[_ ]")
        pattern = re.compile(
            rf"(?im)^\s*\*{{0,2}}{flexible_field}\*{{0,2}}\s*:\s*(.+?)\s*$"
        )
        matches = pattern.findall(field_text)
        if not matches:
            continue

        value = matches[-1].strip()
        field_boxed = extract_boxed_answer(value)
        if field_boxed is not None:
            return field_boxed

        value = value.strip("$ ")
        value = value.rstrip("。.,;；")
        if value:
            return value

    return boxed


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None

    answer = answer.strip()

    # MATH 标准答案常省略这些展示性标记；它们不改变数学值。
    answer = re.sub(
        r"(?<=\d)\s*\^\s*\{\s*\\(?:text|mathrm)\s*"
        r"\{\s*(?:st|nd|rd|th)\s*\}\s*\}",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", answer)
    answer = answer.replace(r"\$", "").replace("$", "")
    answer = remove_text_units(answer)

    answer = answer.replace(r"\left", "")
    answer = answer.replace(r"\right", "")
    answer = answer.replace(r"\dfrac", r"\frac")
    answer = answer.replace(r"\tfrac", r"\frac")
    for spacing in (r"\,", r"\!", r"\;", r"\:"):
        answer = answer.replace(spacing, "")

    # 统一 MATH 中常见的简单分数写法：\frac14、\frac 59、\frac{5}{9}。
    answer = re.sub(
        r"\\frac\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}",
        r"\1/\2",
        answer,
    )
    answer = re.sub(
        r"\\frac\s*\{\s*([+-]?\d+)\s*\}\s*([+-]?\d)",
        r"\1/\2",
        answer,
    )
    answer = re.sub(
        r"\\frac\s*([+-]?\d)\s*\{\s*([+-]?\d+)\s*\}",
        r"\1/\2",
        answer,
    )
    answer = re.sub(
        r"\\frac\s*([+-]?\d)\s*([+-]?\d)",
        r"\1/\2",
        answer,
    )

    answer = re.sub(r"\s+", "", answer)
    answer = re.sub(r"(?<![\d])\.(?=\d)", "0.", answer)
    previous = None
    while answer != previous:
        previous = answer
        answer = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", answer)
    answer = answer.rstrip("。.,;；")

    return answer


def _latex_to_safe_expression(answer: str) -> str | None:
    """Convert a small numeric-LaTeX subset to a safely evaluable expression."""

    expression = answer
    if any(marker in expression for marker in (r"\pm", "=", ",", r"\begin")):
        return None

    # Resolve innermost square roots and fractions first.
    previous = None
    while expression != previous:
        previous = expression
        expression = re.sub(
            r"\\sqrt\{([^{}]+)\}",
            r"sqrt(\1)",
            expression,
        )
        expression = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
            r"((\1)/(\2))",
            expression,
        )

    expression = expression.replace(r"\pi", "pi")
    expression = expression.replace(r"\cdot", "*").replace(r"\times", "*")
    expression = expression.replace("^", "**")
    expression = expression.replace("{", "(").replace("}", ")")
    expression = re.sub(r"(?<=\d)(?=(?:sqrt|pi|\())", "*", expression)
    expression = re.sub(r"(?<=\))(?=(?:\d|sqrt|pi|\())", "*", expression)

    if re.search(r"[^0-9+\-*/().sqrtpi]", expression):
        return None
    return expression


def _evaluate_safe_numeric(expression: str) -> Any | None:
    try:
        import sympy

        tree = ast.parse(expression, mode="eval")

        def evaluate(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return sympy.Rational(str(node.value))
            if isinstance(node, ast.Name) and node.id == "pi":
                return sympy.pi
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left = evaluate(node.left)
                right = evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left**right
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "sqrt"
                and len(node.args) == 1
                and not node.keywords
            ):
                return sympy.sqrt(evaluate(node.args[0]))
            raise ValueError("unsupported expression")

        return evaluate(tree)
    except (ImportError, SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return None


def _scalar_equivalent(first: str, second: str) -> bool:
    if first == second:
        return True
    first_expression = _latex_to_safe_expression(first)
    second_expression = _latex_to_safe_expression(second)
    if first_expression is None or second_expression is None:
        return False
    first_value = _evaluate_safe_numeric(first_expression)
    second_value = _evaluate_safe_numeric(second_expression)
    if first_value is None or second_value is None:
        return False
    try:
        import sympy

        return bool(sympy.simplify(first_value - second_value) == 0)
    except (ImportError, TypeError, ValueError):
        return False


def _split_collection(answer: str) -> tuple[list[str], bool] | None:
    """Return collection components and whether their order is meaningful."""

    value = answer
    if r"\begin{pmatrix}" in value and r"\end{pmatrix}" in value:
        start = value.index(r"\begin{pmatrix}")
        end = value.rindex(r"\end{pmatrix}")
        scalar = value[:start]
        matrix = value[start + len(r"\begin{pmatrix}") : end]
        parts = [item for item in matrix.split(r"\\") if item]
        if scalar:
            parts = [f"({scalar})*({item})" for item in parts]
        return parts, True

    # Expand a scalar a \pm b into its two unordered values.
    plus_minus = re.fullmatch(r"(.+?)\\pm(.+)", value)
    if plus_minus:
        left, right = plus_minus.groups()
        return [f"({left})+({right})", f"({left})-({right})"], False

    is_set = (
        (value.startswith(r"\{") and value.endswith(r"\}"))
        or (value.startswith("{") and value.endswith("}"))
    )
    if is_set:
        value = value[2:-2] if value.startswith(r"\{") else value[1:-1]

    is_tuple = value.startswith("(") and value.endswith(")")
    if is_tuple:
        value = value[1:-1]

    if "," not in value:
        return None
    parts = [part for part in value.split(",") if part]
    return parts, is_tuple


def _collection_equivalent(
    first: tuple[list[str], bool],
    second: tuple[list[str], bool],
) -> tuple[bool, str | None]:
    first_parts, first_ordered = first
    second_parts, second_ordered = second
    if len(first_parts) != len(second_parts):
        return False, None

    if first_ordered or second_ordered:
        matched = all(
            _scalar_equivalent(left, right)
            for left, right in zip(first_parts, second_parts)
        )
        return matched, "ordered_collection" if matched else None

    unmatched = list(second_parts)
    for left in first_parts:
        for index, right in enumerate(unmatched):
            if _scalar_equivalent(left, right):
                unmatched.pop(index)
                break
        else:
            return False, None
    return True, "unordered_collection"


def equivalence_reason(
    predicted: str | None,
    reference: str | None,
) -> str | None:
    if predicted is None or reference is None:
        return None

    normalized_predicted = normalize_answer(predicted)
    normalized_reference = normalize_answer(reference)
    if normalized_predicted is None or normalized_reference is None:
        return None
    if normalized_predicted == normalized_reference:
        return "normalized_exact"
    if _scalar_equivalent(normalized_predicted, normalized_reference):
        return "scalar_numeric"

    predicted_collection = _split_collection(normalized_predicted)
    reference_collection = _split_collection(normalized_reference)
    if predicted_collection is not None and reference_collection is not None:
        equivalent, reason = _collection_equivalent(
            predicted_collection,
            reference_collection,
        )
        if equivalent:
            return reason
    return None


def exact_match(
    predicted: str | None,
    reference: str | None,
) -> bool:
    """Benchmark-safe equivalence for formatting and simple numeric forms."""

    return equivalence_reason(predicted, reference) is not None

def remove_units(answer: str) -> str:
    """
    去除MATH答案中常见的文字单位。

    示例：
    864 \\mbox{ inches}^2 -> 864
    """

    answer = re.sub(
        r"\\(?:mbox|text|mathrm)\s*\{[^{}]*\}",
        "",
        answer,
    )

    answer = re.sub(
        r"\b(?:square|cubic)?\s*"
        r"(?:inches?|feet|yards?|meters?|centimeters?|"
        r"miles?|degrees?)\b",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    # 去掉单位残留的平方或立方
    answer = re.sub(
        r"\^\s*\{?\s*[23]\s*\}?",
        "",
        answer,
    )

    return answer
