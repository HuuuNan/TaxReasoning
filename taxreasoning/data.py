"""Loading the TaxReasoning benchmark and its regulation corpus."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

DEFAULT_QUESTIONS = DATA_DIR / "data-validation_set-V1.xlsx"
DEFAULT_REGULATIONS = DATA_DIR / "regulation_content.xlsx"

# Columns expected in the question workbook.
COL_QUESTION = "Case"
COL_SOLUTION = "Detail_Solution"
COL_ANSWER = "Answer"
COL_TITLES = "Regulation_Title"

_TRAILING_PARENS = re.compile(r"(（[^（）]*）|\([^()]*\))\s*$")


def normalize_title(title: str) -> str:
    """Canonicalise a regulation title for dictionary lookup.

    Titles are cited inconsistently across the two workbooks: some carry a
    trailing revision marker such as ``(2018修订)`` and some mix half- and
    full-width parentheses. Stripping the trailing parenthetical and folding
    parentheses to full width makes the two sides line up.
    """
    cleaned = _TRAILING_PARENS.sub("", str(title))
    return cleaned.replace("(", "（").replace(")", "）").strip()


def load_questions(path: Path | str = DEFAULT_QUESTIONS) -> pd.DataFrame:
    """Load the benchmark questions.

    Returns a frame with the ``Case`` / ``Detail_Solution`` / ``Answer`` /
    ``Regulation_Title`` columns; ``Regulation_Title`` holds one or more
    newline-separated regulation titles per question.
    """
    df = pd.read_excel(path)
    missing = {COL_QUESTION, COL_ANSWER, COL_TITLES} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {sorted(missing)}")
    return df


def load_regulations(path: Path | str = DEFAULT_REGULATIONS) -> dict[str, str]:
    """Load the regulation corpus as ``{normalised title: full text}``.

    Duplicate titles keep their first occurrence, matching the behaviour of the
    original scripts.
    """
    df = pd.read_excel(path)
    if "regulation_title" not in df.columns or "content" not in df.columns:
        raise ValueError(
            f"{path}: expected 'regulation_title' and 'content' columns, "
            f"found {list(df.columns)}"
        )

    corpus: dict[str, str] = {}
    for title, content in zip(df["regulation_title"], df["content"]):
        key = normalize_title(title)
        if key in corpus:
            logger.warning("duplicate regulation title, keeping first: %s", key)
            continue
        corpus[key] = content
    return corpus


def split_titles(cell: str) -> list[str]:
    """Split a ``Regulation_Title`` cell into individual titles."""
    return [line.strip() for line in str(cell).strip().split("\n") if line.strip()]


def build_knowledge(cell: str, corpus: dict[str, str]) -> tuple[str, list[str]]:
    """Concatenate the gold regulation texts cited by one question.

    Returns ``(knowledge_text, unresolved_titles)``. Unlike the original
    scripts -- which indexed the dictionary directly and died with a
    ``KeyError`` on the first citation absent from the corpus -- a missing
    title is reported and skipped so a run can complete over the partial
    corpus. Check the unresolved count before trusting oracle-setting numbers.
    """
    texts: list[str] = []
    unresolved: list[str] = []
    for title in split_titles(cell):
        if title in corpus:
            texts.append(corpus[title])
        elif normalize_title(title) in corpus:
            texts.append(corpus[normalize_title(title)])
        else:
            unresolved.append(title)
    return "\n\n".join(texts), unresolved
