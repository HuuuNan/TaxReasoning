#!/usr/bin/env python3
"""Calibrate accuracy with an external calculator (paper, Figure 2).

    To distinguish between reasoning failures and pure computation errors, we
    follow Zhao et al. and implement an external calculator pipeline.
    Specifically, we use GPT-4o to extract the final mathematical expression
    from each model's CoT output, and then evaluate the expression using an
    external symbolic execution engine.

The gap between the original accuracy and the calibrated accuracy is the share
of questions where the model reasoned correctly but computed wrongly.

    python scripts/run_calculator.py --pred results/deepseek-chat-cot.jsonl

Writes a ``*-calc.jsonl`` plus a summary reporting both accuracies and how many
questions the calculator rescued or broke.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxreasoning import answer as answer_mod  # noqa: E402
from taxreasoning import llm  # noqa: E402

logger = logging.getLogger("run_calculator")

EXTRACT_EXPR_SYSTEM = (
    "你是一个数学表达式抽取助手。给定一段税务计算的解题过程，"
    "抽取出计算最终答案所用的完整数学表达式。要求：\n"
    "1. 只返回一个可直接求值的算术表达式，不要包含等号、单位、变量名或说明文字；\n"
    "2. 保留原始的数值和运算顺序，必要时补上括号以保证运算优先级正确；\n"
    "3. 百分数请写成小数（如 35% 写作 0.35）；\n"
    "4. 如果无法抽取出表达式，只返回 NONE。"
)


def evaluate_expression(expression: str) -> float | None:
    """Evaluate an arithmetic expression with SymPy.

    SymPy parses and evaluates symbolically rather than through ``eval``, so a
    malformed or malicious extraction cannot execute arbitrary code.
    """
    text = expression.strip().strip("`").replace("×", "*").replace("÷", "/").replace("−", "-")
    text = text.replace(",", "").replace("¥", "").replace("￥", "")
    if not text or text.upper() == "NONE":
        return None
    try:
        from sympy import sympify

        value = sympify(text, evaluate=True)
        return float(value.evalf())
    except Exception:  # noqa: BLE001 - extracted expressions are often malformed
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="CoT prediction JSONL")
    parser.add_argument("--expr-model", default="gpt-4o-2024-08-06",
                        help="Model that extracts the expression (paper: GPT-4o)")
    parser.add_argument("--extractor-model", default="gpt-4o-mini",
                        help="Model that extracts the plain final answer, for the baseline number")
    parser.add_argument("--provider", choices=list(llm.PROVIDERS))
    parser.add_argument("--rel-tol", type=float, default=answer_mod.DEFAULT_REL_TOL)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(args.pred, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    expr_client = llm.make_client(args.expr_model, args.provider)
    ans_client = llm.make_client(args.extractor_model, args.provider)

    out_path = args.out or args.pred.with_name(args.pred.stem + "-calc.jsonl")
    base_correct = calc_correct = rescued = broken = 0

    with open(out_path, "w", encoding="utf-8") as handle:
        for record in tqdm(records, desc="Calculator"):
            gold = answer_mod.parse_number(record["gold"])

            # Baseline: the number the model itself stated.
            stated = answer_mod.extract_answer(ans_client, args.extractor_model, record["prediction"])
            base_ok = answer_mod.numbers_match(gold, stated, args.rel_tol)

            # Calibrated: re-evaluate the model's own expression externally.
            expression = llm.chat(
                expr_client,
                args.expr_model,
                [
                    {"role": "system", "content": EXTRACT_EXPR_SYSTEM},
                    {"role": "user", "content": record["prediction"]},
                ],
                temperature=0,
            ).strip()
            computed = evaluate_expression(expression)
            calc_ok = answer_mod.numbers_match(gold, computed, args.rel_tol)

            base_correct += base_ok
            calc_correct += calc_ok
            rescued += calc_ok and not base_ok
            broken += base_ok and not calc_ok

            record.update(
                {
                    "gold_value": gold,
                    "stated_value": stated,
                    "expression": expression,
                    "computed_value": computed,
                    "correct": base_ok,
                    "correct_with_calculator": calc_ok,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(records)
    summary = {
        "pred_file": str(args.pred),
        "model": records[0].get("model"),
        "mode": records[0].get("mode"),
        "n": n,
        "accuracy": base_correct / n,
        "accuracy_with_calculator": calc_correct / n,
        "delta": (calc_correct - base_correct) / n,
        "rescued_by_calculator": rescued,
        "broken_by_calculator": broken,
    }
    (out_path.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("original   accuracy: %.2f%%", 100 * summary["accuracy"])
    logger.info("calibrated accuracy: %.2f%%", 100 * summary["accuracy_with_calculator"])
    logger.info("%d rescued (reasoning right, arithmetic wrong), %d broken by extraction",
                rescued, broken)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
