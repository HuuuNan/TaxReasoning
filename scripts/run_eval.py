#!/usr/bin/env python3
"""Score TaxReasoning predictions.

Implements the paper's protocol ("Answer Extraction and Accuracy Evaluation"):

* **PoT** predictions -- execute the generated ``solution()`` and take its
  return value.
* **Everything else** -- GPT-4o-mini extracts the final numerical answer from
  the reasoning trace.
* Either way, the extracted number is compared to the gold answer by **exact
  numerical matching with a +/-0.2% tolerance**.

The comparison is arithmetic, not a yes/no judgement by an LLM, so scoring a
fixed set of predictions is deterministic apart from the extraction step.

Examples::

    python scripts/run_eval.py --pred results/gpt-4o-2024-08-06-oracle.jsonl
    python scripts/run_eval.py --pred results/deepseek-chat-pot.jsonl --allow-exec

``--scoring`` defaults to ``program`` for files whose ``mode`` is ``pot``, and
to ``answer`` otherwise.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import signal
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxreasoning import answer as answer_mod  # noqa: E402
from taxreasoning import llm  # noqa: E402

logger = logging.getLogger("run_eval")

CODE_BLOCK_PATTERN = re.compile(
    r"(?:```(?:python)?|'''(?:python)?)\s*([\s\S]*?)(?:```|'''|$)", re.DOTALL
)


def extract_code_block(text: str) -> str:
    match = CODE_BLOCK_PATTERN.search(text)
    return match.group(1).strip() if match else text


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout


def run_solution(text: str, timeout: int = 10):
    """Execute the generated ``solution()`` and return its value, or ``None``.

    The function body is isolated from the surrounding response via the AST, so
    only the definition and a single call are compiled. This is *not* a
    sandbox -- the code runs with full process privileges. The timeout stops
    infinite loops, not filesystem or network access.
    """
    if not isinstance(text, str):
        return None
    code_content = extract_code_block(text)
    try:
        module = ast.parse(code_content)
        func_def = next(
            (n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "solution"),
            None,
        )
        if not func_def:
            return None

        call_node = ast.parse("result = solution()").body[0]
        exec_module = ast.Module(body=[func_def, call_node], type_ignores=[])
        code_obj = compile(exec_module, "<generated>", "exec")

        local_vars: dict = {}
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
        try:
            exec(code_obj, {}, local_vars)  # noqa: S102 - deliberate, see docstring
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        return local_vars.get("result")
    except Exception:  # noqa: BLE001 - generated code fails in arbitrary ways
        return None


def load_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="Prediction JSONL from run_predict.py")
    parser.add_argument("--scoring", choices=["answer", "program"],
                        help="Default: program for pot predictions, answer otherwise")
    parser.add_argument("--extractor-model", default="gpt-4o-mini",
                        help="Model that extracts the final number (paper: GPT-4o-mini)")
    parser.add_argument("--provider", choices=list(llm.PROVIDERS), help="Override extractor provider routing")
    parser.add_argument("--rel-tol", type=float, default=answer_mod.DEFAULT_REL_TOL,
                        help="Relative tolerance for numerical matching (paper: 0.002)")
    parser.add_argument("--out", type=Path, help="Output JSONL (default: <pred stem>-eval.jsonl)")
    parser.add_argument("--allow-exec", action="store_true",
                        help="Required for program scoring: executes model-generated Python in this process")
    parser.add_argument("--exec-timeout", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    records = load_predictions(args.pred)
    if not records:
        parser.error(f"{args.pred} is empty")

    scoring = args.scoring or ("program" if records[0].get("mode") in ("pot", "pal") else "answer")
    logger.info("scoring=%s (mode=%s) rel_tol=%.4f", scoring, records[0].get("mode"), args.rel_tol)

    if scoring == "program" and not args.allow_exec:
        parser.error(
            "program scoring runs Python written by the evaluated model in this "
            "process. Pass --allow-exec to confirm, ideally inside a container."
        )

    out_path = args.out or args.pred.with_name(args.pred.stem + "-eval.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = None if scoring == "program" else llm.make_client(args.extractor_model, args.provider)

    correct = 0
    no_answer = 0

    with open(out_path, "w", encoding="utf-8") as handle:
        for record in tqdm(records, desc="Scoring"):
            if scoring == "program":
                predicted = run_solution(record["prediction"], args.exec_timeout)
            else:
                predicted = answer_mod.extract_answer(client, args.extractor_model, record["prediction"])

            gold_value = answer_mod.parse_number(record["gold"])
            is_correct = answer_mod.numbers_match(gold_value, predicted, args.rel_tol)

            record["gold_value"] = gold_value
            record["predicted_value"] = None if predicted is None else float(predicted) if isinstance(predicted, (int, float)) else str(predicted)
            record["correct"] = is_correct
            correct += is_correct
            no_answer += predicted is None
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    accuracy = correct / len(records)
    logger.info("accuracy: %d/%d = %.2f%%", correct, len(records), 100 * accuracy)
    if no_answer:
        logger.info("%d predictions yielded no number and were scored wrong", no_answer)

    summary = {
        "pred_file": str(args.pred),
        "model": records[0].get("model"),
        "mode": records[0].get("mode"),
        "scoring": scoring,
        "extractor_model": None if scoring == "program" else args.extractor_model,
        "rel_tol": args.rel_tol,
        "n": len(records),
        "correct": correct,
        "accuracy": accuracy,
        "no_answer": no_answer,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s and %s", out_path, summary_path)


if __name__ == "__main__":
    main()
