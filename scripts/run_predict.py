#!/usr/bin/env python3
"""Generate model predictions for the TaxReasoning benchmark.

Covers all seven methods reported in Table 2 of the paper:

    Vanilla          cot, pot
    Single-turn RAG  standard, summary
    Multi-turn RAG   interact, inlinesearch
    Oracle           oracle

Examples
--------
Vanilla settings, no retrieval::

    python scripts/run_predict.py --model gpt-4o-2024-08-06 --mode cot
    python scripts/run_predict.py --model gpt-4o-2024-08-06 --mode pot

Oracle -- gold regulations annotated with each question::

    python scripts/run_predict.py --model deepseek-chat --mode oracle

Retrieval settings (build the index first: ``python scripts/build_index.py``)::

    python scripts/run_predict.py --model deepseek-chat --mode standard
    python scripts/run_predict.py --model deepseek-chat --mode summary
    python scripts/run_predict.py --model deepseek-chat --mode interact
    python scripts/run_predict.py --model deepseek-chat --mode inlinesearch

Output is JSONL, one record per question, appended as it is produced.
Re-running the same command resumes: questions already in the output file are
skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxreasoning import data, llm, methods, prompts  # noqa: E402

logger = logging.getLogger("run_predict")

VANILLA_MODES = ("cot", "pot")
RETRIEVAL_MODES = ("standard", "summary", "interact", "inlinesearch")
ALL_MODES = VANILLA_MODES + ("oracle",) + RETRIEVAL_MODES


def load_examples(args) -> list[dict]:
    """Load benchmark questions, attaching gold regulations in oracle mode."""
    df = data.load_questions(args.questions)
    corpus = data.load_regulations(args.regulations) if args.mode == "oracle" else {}

    examples = []
    total_unresolved = 0
    for idx, row in df.iterrows():
        knowledge, unresolved = "", []
        if args.mode == "oracle":
            knowledge, unresolved = data.build_knowledge(row[data.COL_TITLES], corpus)
            total_unresolved += len(unresolved)
        examples.append(
            {
                "id": int(idx),
                "question": row[data.COL_QUESTION],
                "gold": row[data.COL_ANSWER],
                "knowledge": knowledge,
                "unresolved_titles": unresolved,
            }
        )

    if total_unresolved:
        logger.warning(
            "%d regulation citations could not be resolved against %s. "
            "Oracle-setting numbers will understate the available context.",
            total_unresolved,
            Path(args.regulations).name,
        )
    return examples


def completed_ids(path: Path) -> set[int]:
    """Ids already present in an output file, for resuming."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def predict_one(args, example, client, summarizer, retriever) -> dict:
    """Run one example under the selected method and return its record."""
    question = example["question"]
    record: dict = {"retrieved": [], "trace": []}

    if args.mode in VANILLA_MODES or args.mode == "oracle":
        messages = prompts.build_messages(args.mode, question, example["knowledge"] or None)
        record["prediction"] = llm.chat(client, args.model, messages)
        return record

    # Every retrieval mode except inlinesearch starts from one up-front query.
    if args.mode != "inlinesearch":
        top_k = args.top_k or (
            methods.DEFAULT_TOP_K_MULTI if args.mode == "interact" else methods.DEFAULT_TOP_K_SINGLE
        )
        documents = retriever.search(question, top_k=top_k)
        record["retrieved"] = [{"id": d["id"], "title": d["title"], "score": d["score"]} for d in documents]

    if args.mode == "standard":
        record["prediction"] = llm.chat(
            client, args.model, methods.build_standard_messages(question, documents)
        )

    elif args.mode == "summary":
        knowledge, summaries = methods.build_summary_knowledge(
            summarizer, args.summarizer_model, question, documents
        )
        record["summaries"] = summaries
        record["prediction"] = llm.chat(
            client, args.model, prompts.build_messages("summary", question, knowledge)
        )

    elif args.mode == "interact":
        # The model first sees summaries; Check reveals the full document.
        _, summaries = methods.build_summary_knowledge(
            summarizer, args.summarizer_model, question, documents
        )
        summary_texts = [
            s["refine"].removeprefix(methods.RELEVANT_PREFIX) for s in summaries
        ]
        record["summaries"] = summaries
        record["prediction"], record["trace"] = methods.run_interact(
            client, args.model, question, documents, summary_texts, args.max_turns
        )

    elif args.mode == "inlinesearch":
        record["prediction"], record["trace"] = methods.run_inlinesearch(
            client, args.model, question, retriever, args.max_turns
        )

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model name as the provider expects it")
    parser.add_argument("--mode", required=True, choices=ALL_MODES, help="Method from Table 2")
    parser.add_argument("--provider", choices=list(llm.PROVIDERS), help="Override provider routing")
    parser.add_argument("--questions", type=Path, default=data.DEFAULT_QUESTIONS)
    parser.add_argument("--regulations", type=Path, default=data.DEFAULT_REGULATIONS)
    parser.add_argument("--index", type=Path, help="Retrieval index (default: index/regulations.npz)")
    parser.add_argument("--top-k", type=int, help="Documents to retrieve (default: 5 single-turn, 10 interact)")
    parser.add_argument("--max-turns", type=int, default=methods.DEFAULT_MAX_TURNS,
                        help="Turn cap for interact / inlinesearch")
    parser.add_argument("--summarizer-model", default="gpt-4o-2024-08-06",
                        help="Model that summarises retrieved documents (paper: GPT-4o)")
    parser.add_argument("--out", type=Path, help="Output JSONL (default: results/<model>-<mode>.jsonl)")
    parser.add_argument("--limit", type=int, help="Only run the first N questions (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    examples = load_examples(args)
    if args.limit:
        examples = examples[: args.limit]

    out_path = args.out or Path("results") / f"{args.model.replace('/', '_')}-{args.mode}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = completed_ids(out_path)
    todo = [ex for ex in examples if ex["id"] not in done]
    logger.info(
        "model=%s mode=%s | %d questions, %d already done, %d to run -> %s",
        args.model, args.mode, len(examples), len(done), len(todo), out_path,
    )
    if not todo:
        return

    client = llm.make_client(args.model, args.provider)

    retriever = None
    if args.mode in RETRIEVAL_MODES:
        from taxreasoning import retrieval

        retriever = retrieval.Retriever.load(args.index or retrieval.DEFAULT_INDEX)
        logger.info("loaded index: %d passages, encoder=%s",
                    len(retriever.passages), retriever.encoder.name)

    summarizer = None
    if args.mode in ("summary", "interact"):
        summarizer = llm.make_client(args.summarizer_model)

    with open(out_path, "a", encoding="utf-8") as handle:
        for example in tqdm(todo, desc=f"Predicting ({args.mode})"):
            record = predict_one(args, example, client, summarizer, retriever)
            record.update(
                {
                    "id": example["id"],
                    "model": args.model,
                    "mode": args.mode,
                    "question": example["question"],
                    "gold": example["gold"],
                    "unresolved_titles": example["unresolved_titles"],
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
