"""Answer extraction and numerical matching.

Implements the protocol from the paper's "Answer Extraction and Accuracy
Evaluation" section:

    For PoT, we directly execute the model-generated Python code to obtain the
    output. For other methods, we use GPT-4o-mini to extract the final
    numerical answer from the generated reasoning trace. Evaluation is based on
    exact numerical matching, with a tolerance of +/-0.2% to account for minor
    rounding discrepancies.

Note this is *not* a yes/no LLM judge. The judge only extracts a number; the
comparison itself is arithmetic, so it is deterministic and reproducible.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: Relative tolerance for numerical matching (0.2%, per the paper).
DEFAULT_REL_TOL = 0.002

EXTRACT_SYSTEM = (
    "你是一个答案抽取助手。从给定的解题过程中抽取最终的数值答案。"
    "只返回一个数字，不要包含单位、货币符号、千分位逗号或任何其他文字。"
    "如果解题过程没有给出明确的最终数值答案，只返回 NONE。"
)

# Chinese numeric-magnitude suffixes, applied to the value that precedes them.
_UNIT_SCALES = [
    ("亿元", 1e8), ("亿", 1e8),
    ("万元", 1e4), ("万", 1e4),
]

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_number(text) -> float | None:
    """Parse a numeric value out of a short answer string.

    Handles the forms the gold answers use -- ``'3240元'``, ``'2,155元'``,
    ``'1.5万元'``, ``'0元'`` -- returning a value in base units (元). Returns
    ``None`` when no number is present.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    s = str(text).strip()
    if not s or s.upper() == "NONE":
        return None

    scale = 1.0
    for suffix, factor in _UNIT_SCALES:
        if suffix in s:
            scale = factor
            s = s.replace(suffix, "")
            break

    match = _NUMBER_RE.search(s)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "")) * scale
    except ValueError:
        return None


def extract_answer(client, model: str, reasoning_trace: str) -> float | None:
    """Ask the extraction model for the final numeric answer of a trace.

    Falls back to a local parse of the last number in the trace if the model
    returns something unparsable, so a flaky extraction call does not silently
    score a correct answer as wrong.
    """
    from taxreasoning import llm

    reply = llm.chat(
        client,
        model,
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": reasoning_trace},
        ],
        temperature=0,
    )
    value = parse_number(reply)
    if value is None and reply.strip().upper() != "NONE":
        logger.debug("extractor returned %r; falling back to local parse", reply)
        value = parse_number(_last_number(reasoning_trace))
    return value


def _last_number(text: str) -> str | None:
    """Last number appearing in a trace -- the fallback for a failed extraction."""
    matches = _NUMBER_RE.findall(str(text))
    return matches[-1] if matches else None


def numbers_match(gold, pred, rel_tol: float = DEFAULT_REL_TOL) -> bool:
    """Compare two numbers with a relative tolerance.

    ``rel_tol`` defaults to the paper's +/-0.2%. When the gold value is zero the
    comparison falls back to an absolute tolerance, since a relative one would
    only ever accept exactly zero.
    """
    gold_val = parse_number(gold) if not isinstance(gold, float) else gold
    pred_val = parse_number(pred) if not isinstance(pred, float) else pred

    if gold_val is None or pred_val is None:
        return False
    if gold_val == 0:
        return abs(pred_val) <= rel_tol
    return abs(pred_val - gold_val) / abs(gold_val) <= rel_tol
