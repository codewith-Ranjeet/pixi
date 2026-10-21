"""Small, safe local tools used by the Pixi terminal chat."""

import ast
import json
import re
from decimal import Decimal, localcontext
from fractions import Fraction


_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "please", "the", "to",
    "what", "which", "who", "why", "with", "would", "you", "your", "tell", "me",
    "give", "name", "state", "briefly", "short", "sentence", "answer", "only",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class CalculatorError(ValueError):
    """Raised when an explicitly requested calculation is invalid or unsafe."""


def extract_arithmetic_expression(query):
    """Return a restricted arithmetic expression, or None if this is not math."""
    text = query.casefold().strip()
    if not text or len(text) > 300:
        return None

    # Safely handle a narrow, explicit subtraction word-problem form. Requiring
    # both an action (ate/lost/etc.) and a remaining-count question avoids
    # treating unrelated statements with two numbers as arithmetic.
    item = r"[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?"
    remaining = re.fullmatch(
        rf"(?:i\s+(?:have|had)|there\s+(?:are|were))\s+({_NUMBER})\s+{item}"
        rf"\s*(?:,?\s+and\s+|,\s*)(?:i\s+)?(?:ate|lost|used|spent|sold|removed|"
        rf"gave\s+away|threw\s+away)\s+({_NUMBER})\b.*"
        rf"\b(?:how\s+many|how\s+much|what)\b.*\b(?:remain(?:ed|s)?|left)\b[?.! ]*",
        text,
    )
    if remaining:
        return f"{remaining.group(1)} - {remaining.group(2)}"

    # Common verbal forms whose operand order is easy to misread.
    match = re.fullmatch(rf"(?:please\s+)?add\s+({_NUMBER})\s+and\s+({_NUMBER})[?.! ]*", text)
    if match:
        return f"{match.group(1)} + {match.group(2)}"
    match = re.fullmatch(rf"(?:please\s+)?subtract\s+({_NUMBER})\s+from\s+({_NUMBER})[?.! ]*", text)
    if match:
        return f"{match.group(2)} - {match.group(1)}"
    match = re.fullmatch(rf"(?:please\s+)?(?:multiply|times)\s+({_NUMBER})\s+by\s+({_NUMBER})[?.! ]*", text)
    if match:
        return f"{match.group(1)} * {match.group(2)}"
    match = re.fullmatch(rf"(?:please\s+)?divide\s+({_NUMBER})\s+by\s+({_NUMBER})[?.! ]*", text)
    if match:
        return f"{match.group(1)} / {match.group(2)}"

    # Normalize a deliberately small set of natural-language operators.
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    # Some users type "time" for "times". Treat it as multiplication only
    # when it sits between numeric operands, so ordinary mentions of time stay text.
    text = re.sub(r"(?<=\d)\s+times?\s+(?=[+-]?\d)", " * ", text)
    text = re.sub(rf"({_NUMBER})\s*(?:percent|per\s+cent|%)", r"(\1 / 100)", text)
    text = re.sub(r"\b(?:multiplied\s+by|times)\b", "*", text)
    text = re.sub(r"\bdivided\s+by\b|\bover\b", "/", text)
    text = re.sub(r"\bplus\b", "+", text)
    text = re.sub(r"\bminus\b", "-", text)
    text = re.sub(r"\bof\b", "*", text)
    text = text.replace("^", "**")

    # Drop only common instruction wrappers; any remaining prose is rejected.
    text = re.sub(
        r"^(?:please\s+)?(?:(?:what\s+is|what's|calculate|compute|evaluate|solve)\s+|"
        r"find\s+(?:the\s+)?(?:(?:answer|result|value)(?:\s+of)?\s*)?|"
        r"answer\s+(?:with\s+)?(?:the\s+)?(?:result|answer)\s*:?\s*)",
        "",
        text,
    )
    text = re.sub(r"[?.!]+$", "", text).strip()
    text = re.sub(r"\s*=+$", "", text).strip()

    # A math operator or explicit arithmetic verb is required. This keeps
    # years, counts, and ordinary factual questions out of the calculator.
    if not re.search(r"(?:\d\s*[+*/%]\s*\d|\d\s*-\s*\d|\d\s*\*\*\s*\d)", text):
        return None
    if not re.fullmatch(r"[\d\s().+*/%\-]+", text):
        return None
    return text


def _checked(value):
    if value.numerator.bit_length() > 256 or value.denominator.bit_length() > 256:
        raise CalculatorError("The result is too large for this calculator.")
    if abs(value) > 10**18:
        raise CalculatorError("The result is outside the supported range.")
    return value


def _evaluate_node(node):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _checked(Fraction(str(node.value)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand)
        return _checked(value if isinstance(node.op, ast.UAdd) else -value)
    if isinstance(node, ast.BinOp):
        left, right = _evaluate_node(node.left), _evaluate_node(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise CalculatorError("Division by zero is undefined.")
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise CalculatorError("Division by zero is undefined.")
            result = Fraction(left // right)
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise CalculatorError("Division by zero is undefined.")
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 12 or abs(left) > 1_000_000:
                raise CalculatorError("Powers are limited to integer exponents from -12 to 12.")
            result = left ** right.numerator
        else:
            raise CalculatorError("That operator is not supported.")
        return _checked(result)
    raise CalculatorError("Only basic arithmetic is supported.")


def safe_calculate(expression):
    """Evaluate arithmetic with an AST whitelist; never execute Python code."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise CalculatorError("I couldn't read that arithmetic expression.") from exc
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise CalculatorError("That expression is too complex.")
    return _evaluate_node(tree)


def format_fraction(value):
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    original_denominator = denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator == 1:
        # Terminating decimal: enough precision to represent it exactly.
        with localcontext() as context:
            context.prec = max(len(str(abs(value.numerator))) + len(str(original_denominator)) + 4, 32)
            rendered = format(Decimal(value.numerator) / Decimal(original_denominator), "f")
        rendered = rendered.rstrip("0").rstrip(".")
        return rendered
    approx = f"{float(value):.10g}"
    return f"{value.numerator}/{value.denominator} (approximately {approx})"


def calculate_query(query):
    """Return (expression, exact result), None when query is not arithmetic."""
    expression = extract_arithmetic_expression(query)
    if expression is None:
        return None
    return expression, safe_calculate(expression)


def _terms(text):
    return {token.lower() for token in _TOKEN_RE.findall(text)} - _STOP_WORDS


def load_fact_store(path):
    """Load and validate a local JSON fact file. No network access is used."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        raise ValueError("Fact file must be a JSON object with a 'facts' list.")
    facts = []
    seen_ids = set()
    for index, entry in enumerate(data["facts"]):
        required = ("id", "question", "answer", "keywords", "source_title", "source_url")
        if not isinstance(entry, dict) or any(key not in entry for key in required):
            raise ValueError(f"Fact entry {index} is missing required fields: {required}")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate fact id: {entry['id']}")
        if not isinstance(entry["keywords"], list) or not entry["keywords"]:
            raise ValueError(f"Fact {entry['id']} must have a non-empty keywords list.")
        if not all(isinstance(entry[key], str) and entry[key].strip() for key in ("id", "question", "answer", "source_title", "source_url")):
            raise ValueError(f"Fact {entry['id']} contains an empty text field.")
        prepared = dict(entry)
        prepared["_terms"] = _terms(" ".join([entry["question"], *entry["keywords"]]))
        facts.append(prepared)
        seen_ids.add(entry["id"])
    return facts


def retrieve_fact(query, facts, threshold=0.68, min_margin=0.08):
    """Return a high-confidence local fact match, otherwise None."""
    query_terms = _terms(query)
    if len(query_terms) < 2 or not facts:
        return None
    ranked = []
    for fact in facts:
        terms = fact["_terms"]
        overlap = query_terms & terms
        if len(overlap) < 2:
            continue
        coverage = len(overlap) / len(query_terms)
        precision = len(overlap) / max(len(terms), 1)
        score = 0.8 * coverage + 0.2 * precision
        ranked.append((score, len(overlap), fact))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not ranked or ranked[0][0] < threshold:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < min_margin:
        return None
    return ranked[0][2]


def looks_like_fact_query(query):
    # These are conversational, advisory, or evaluative prompts. They should
    # still reach the model even though they begin with a question word.
    if re.match(
        r"^\s*(?:who\s+are\s+you|what\s+(?:can|do)\s+you\s+|"
        r"what\s+(?:are|is)\s+(?:some|a\s+few|the\s+best\s+way|your\s+opinion)|"
        r"what\s+do\s+you\s+think|why\s+should\s+i|how\s+can\s+i)\b",
        query,
        re.IGNORECASE,
    ):
        return False
    if re.match(r"^\s*(?:capital|population|currency|language)\s+of\s+[a-z]", query, re.IGNORECASE):
        return True
    return bool(re.match(
        r"^\s*(?:who|what|where|when|which|why|how\s+(?:many|much|often|far|old)|"
        r"name\s+(?:the|a|one)|tell\s+me\s+(?:the|about))\b",
        query,
        re.IGNORECASE,
    ))


def get_builtin_response(query):
    """Return a short deterministic response for a few common chat intents."""
    text = re.sub(r"[\s.!?,]+$", "", query.casefold().strip())
    if re.fullmatch(r"(?:hi|hello|hey|hiya)(?:\s+there)?", text):
        return "Hi! I'm Pixi. I can do basic arithmetic and answer questions covered by my local sourced fact list."
    if re.fullmatch(r"(?:who are you|what are you)", text) or re.fullmatch(
        r"(?:hi|hello|hey)[,! ]+(?:who are you|what are you)", text
    ):
        return "I'm Pixi, a small experimental language model. I can calculate basic arithmetic and look up facts in my local sourced list."
    if "if you don't know" in text or "if you do not know" in text:
        return "If I don't have a verified local fact or a reliable calculation, I should say I don't know rather than guess."
    if "?" not in query and re.match(r"^i\s+have\s+", text):
        return "Got it."
    return None
