#!/usr/bin/env python3
"""Check that the benchmark questions and the regulation corpus line up.

Every question cites one or more regulation titles, and the oracle setting
feeds the matching regulation text to the model. If a cited title is absent
from ``regulation_content.xlsx`` the model silently gets less context than the
setting claims, so run this before trusting any oracle-setting number.

    python tools/check_data.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxreasoning import data  # noqa: E402


def main() -> int:
    questions = data.load_questions()
    corpus = data.load_regulations()

    print(f"questions:   {len(questions)} rows from {data.DEFAULT_QUESTIONS.name}")
    print(f"regulations: {len(corpus)} unique titles from {data.DEFAULT_REGULATIONS.name}")

    resolved = 0
    unresolved: Counter[str] = Counter()
    questions_hit = 0

    for cell in questions[data.COL_TITLES]:
        _, missing = data.build_knowledge(cell, corpus)
        cited = len(data.split_titles(cell))
        resolved += cited - len(missing)
        unresolved.update(missing)
        if missing:
            questions_hit += 1

    total = resolved + sum(unresolved.values())
    print(f"\ncitations:   {resolved}/{total} resolved ({100 * resolved / total:.1f}%)")
    print(f"questions with at least one missing regulation: {questions_hit}/{len(questions)}")

    if unresolved:
        print(f"\n{len(unresolved)} distinct titles are cited but absent from the corpus:")
        for title, count in unresolved.most_common():
            print(f"  [{count:2d}x] {title}")
        return 1

    print("\nAll cited regulations are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
