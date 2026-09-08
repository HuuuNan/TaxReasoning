#!/usr/bin/env python3
"""Win-Tie-Lose comparison between two methods (paper, Figure 3).

    "Win" means the first method is correct and the second is not; "Lose"
    means the reverse; "Tie" means both are correct or both incorrect.

The paper compares STANDARD vs SUMMARY and INTERACT vs INLINESEARCH, using
DeepSeek-R1 as the case study::

    python scripts/analyze_wtl.py \\
        --a results/deepseek-reasoner-standard-eval.jsonl \\
        --b results/deepseek-reasoner-summary-eval.jsonl

    python scripts/analyze_wtl.py \\
        --a results/deepseek-reasoner-interact-eval.jsonl \\
        --b results/deepseek-reasoner-inlinesearch-eval.jsonl

Takes the ``*-eval.jsonl`` files produced by ``run_eval.py`` and compares them
question by question, so the two runs must cover the same question ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_scored(path: Path) -> dict[int, bool]:
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    missing = [r for r in records if "correct" not in r]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} records have no 'correct' field -- run run_eval.py first")
    return {r["id"]: bool(r["correct"]) for r in records}


def label(path: Path, scored_path: Path) -> str:
    """Name a run by its mode/model, falling back to the filename."""
    with open(scored_path, encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    mode, model = first.get("mode"), first.get("model")
    return f"{mode} ({model})" if mode and model else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="First method's scored predictions")
    parser.add_argument("--b", type=Path, required=True, help="Second method's scored predictions")
    parser.add_argument("--out", type=Path, help="Write the comparison as JSON")
    args = parser.parse_args()

    a, b = load_scored(args.a), load_scored(args.b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("the two files share no question ids")
    if len(shared) < max(len(a), len(b)):
        print(f"note: comparing the {len(shared)} ids present in both files "
              f"({len(a)} in A, {len(b)} in B)\n")

    win = [i for i in shared if a[i] and not b[i]]
    lose = [i for i in shared if b[i] and not a[i]]
    tie_both = [i for i in shared if a[i] and b[i]]
    tie_neither = [i for i in shared if not a[i] and not b[i]]

    name_a, name_b = label(args.a, args.a), label(args.b, args.b)
    n = len(shared)

    print(f"A = {name_a}")
    print(f"B = {name_b}")
    print(f"n = {n}\n")
    print(f"  Win  (A right, B wrong): {len(win):4d}  {100 * len(win) / n:5.1f}%")
    print(f"  Tie  (both right)      : {len(tie_both):4d}  {100 * len(tie_both) / n:5.1f}%")
    print(f"  Tie  (both wrong)      : {len(tie_neither):4d}  {100 * len(tie_neither) / n:5.1f}%")
    print(f"  Lose (B right, A wrong): {len(lose):4d}  {100 * len(lose) / n:5.1f}%")
    print(f"\n  accuracy A: {100 * sum(a[i] for i in shared) / n:.2f}%")
    print(f"  accuracy B: {100 * sum(b[i] for i in shared) / n:.2f}%")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "a": {"file": str(args.a), "label": name_a},
                    "b": {"file": str(args.b), "label": name_b},
                    "n": n,
                    "win": len(win), "lose": len(lose),
                    "tie_both_correct": len(tie_both), "tie_both_wrong": len(tie_neither),
                    "win_ids": win, "lose_ids": lose,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
