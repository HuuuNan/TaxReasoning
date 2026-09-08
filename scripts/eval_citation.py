#!/usr/bin/env python3
"""Citation / attribution metrics, adapted from ALCE.

Derived from https://github.com/princeton-nlp/ALCE -- ``eval.py``
Copyright (c) 2023 Princeton Natural Language Processing, MIT License.
See the NOTICE file at the repository root.

Changes from upstream ALCE:

* The NLI entailment model (TRUE / T5-11B) and MAUVE are replaced with an LLM
  judge, as in the paper's baselines -- ``compute_autoais``, ``compute_claims``
  and ``compute_mauve`` became ``*_with_api`` variants.
* Ported from the pre-1.0 ``openai.ChatCompletion`` interface to the shared
  client in ``taxreasoning.llm``; the API key comes from the environment.
* ``compute_qampari_f1`` and the local-model code paths were dropped; cache
  locations come from the environment rather than being hardcoded.
* ``normalize_answer`` / ``remove_citations`` are vendored in
  ``taxreasoning/alce_utils.py`` rather than imported from ALCE's ``utils``.

This operates on **ALCE-format** files (``{"data": [{"question", "output",
"answer", "docs", ...}]}``), not on the JSONL that ``run_predict.py`` writes.
It scores the citation baselines, not the numeric tax answers -- use
``run_eval.py`` for those.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxreasoning import llm  # noqa: E402
from taxreasoning.alce_utils import normalize_answer, remove_citations  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENTAILMENT_SYSTEM = "Determine if the claim is entailed by the context. Answer only 'yes' or 'no'."


def _sent_tokenize(text: str) -> list[str]:
    """Split into sentences, downloading the NLTK model on first use."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
    return nltk.sent_tokenize(text)


def compute_f1(a_gold: str, a_pred: str) -> float:
    def _get_tokens(s):
        return normalize_answer(s).split() if s else []

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context) -> bool:
    n_context = normalize_answer(context)
    return any(normalize_answer(sa) in n_context for sa in short_answers)


def compute_len(data) -> float:
    return sum(len(item["output"].split()) for item in data) / len(data)


def compute_str_em(data):
    if "qa_pairs" not in data[0] or data[0]["qa_pairs"] is None:
        return 0, 0

    acc, hit = [], []
    for item in data:
        loc_acc = [
            exact_presence(qa_pair["short_answers"], item["output"])
            for qa_pair in item["qa_pairs"]
        ]
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))
    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_rouge(data) -> float:
    from rouge_score import rouge_scorer, scoring

    metrics = ["rougeLsum"]
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    hypotheses, references1, references2 = [], [], []
    for item in data:
        hypotheses.append(item["output"])
        if item.get("annotations"):  # ASQA carries two reference long answers
            references1.append(item["annotations"][0]["long_answer"])
            references2.append(item["annotations"][1]["long_answer"])
        else:
            references1.append(item["answer"])
            references2.append(item["answer"])

    def _prep(texts):
        return ["\n".join(_sent_tokenize(t.lower())) for t in texts]

    hypotheses, references1, references2 = _prep(hypotheses), _prep(references1), _prep(references2)

    for hyp, ref1, ref2 in zip(hypotheses, references1, references2):
        scores1 = scorer.score(ref1, hyp)
        scores2 = scorer.score(ref2, hyp)
        aggregator.add_scores(
            scores1 if scores1["rougeLsum"].fmeasure > scores2["rougeLsum"].fmeasure else scores2
        )

    return 100 * aggregator.aggregate()["rougeLsum"].mid.fmeasure


def _entails(client, model: str, context: str, claim: str) -> bool:
    reply = llm.chat(
        client,
        model,
        [
            {"role": "system", "content": ENTAILMENT_SYSTEM},
            {"role": "user", "content": f"Context: {context}\nClaim: {claim}\nEntailed?"},
        ],
        max_tokens=3,
        temperature=0,
    )
    return "yes" in reply.strip().lower()


def compute_claims_with_api(data, client, model: str) -> float:
    logger.info("Computing claims with API...")
    scores = []
    for item in tqdm(data):
        if not item.get("claims"):
            continue
        normalized_output = remove_citations(item["output"])
        entail = sum(_entails(client, model, normalized_output, c) for c in item["claims"])
        scores.append(entail / len(item["claims"]))

    if not scores:
        logger.warning("No items with a 'claims' field; skipping claims evaluation")
        return 0
    return 100 * np.mean(scores)


def compute_autoais_with_api(data, client, model: str, at_most_citations: int = 3) -> dict:
    logger.info("Computing AutoAIS with API...")
    citation_rec_scores, citation_prec_scores = [], []

    for item in tqdm(data):
        sents = _sent_tokenize(item["output"])
        if not sents:
            continue
        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail, entail_prec, total_citations = 0, 0, 0

        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]
            ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]
            if not ref:
                continue
            if at_most_citations is not None:
                ref = ref[:at_most_citations]
            total_citations += len(ref)

            context = "\n\n".join(
                f"Title: {item['docs'][i]['title']}\n{item['docs'][i]['text']}"
                for i in ref
                if i < len(item["docs"])
            )

            joint_entail = int(_entails(client, model, context, target_sent))
            entail += joint_entail

            if joint_entail and len(ref) > 1:
                # Precision: does each cited passage individually support the sentence?
                for i in ref:
                    if i < len(item["docs"]):
                        passage = f"Title: {item['docs'][i]['title']}\n{item['docs'][i]['text']}"
                        entail_prec += int(_entails(client, model, passage, target_sent))
            else:
                entail_prec += joint_entail

        citation_rec_scores.append(entail / len(sents))
        citation_prec_scores.append(entail_prec / total_citations if total_citations else 0)

    return {
        "citation_rec": 100 * np.mean(citation_rec_scores),
        "citation_prec": 100 * np.mean(citation_prec_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--f", type=str, required=True, help="ALCE-format output file to score")
    parser.add_argument("--no_rouge", action="store_true", help="Do not evaluate ROUGE score")
    parser.add_argument("--citations", action="store_true", help="Evaluate citation recall/precision")
    parser.add_argument("--claims_nli", action="store_true", help="Evaluate claim entailment (ELI5)")
    parser.add_argument("--at_most_citations", type=int, default=3, help="Cap citations per sentence")
    parser.add_argument("--judge-model", default="gpt-4o-mini-2024-07-18", help="LLM used for entailment")
    parser.add_argument("--provider", choices=list(llm.PROVIDERS), help="Override judge provider routing")
    args = parser.parse_args()

    with open(args.f, encoding="utf-8") as handle:
        data = json.load(handle)["data"]

    logger.warning("Truncating each output at the first newline and stripping <|im_end|>.")
    for item in data:
        item["output"] = item["output"].strip().split("\n")[0].replace("<|im_end|>", "")

    normalized_data = copy.deepcopy(data)
    for item in normalized_data:
        item["output"] = remove_citations(item["output"])

    result = {"length": compute_len(normalized_data)}
    result["str_em"], result["str_hit"] = compute_str_em(normalized_data)

    if not args.no_rouge:
        result["rougeLsum"] = compute_rouge(normalized_data)

    if args.citations or args.claims_nli:
        client = llm.make_client(args.judge_model, args.provider)
        if args.citations:
            result.update(
                compute_autoais_with_api(data, client, args.judge_model, args.at_most_citations)
            )
        if args.claims_nli:
            result["claims_nli"] = compute_claims_with_api(
                normalized_data, client, args.judge_model
            )

    print(result)
    with open(args.f + ".score", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4)


if __name__ == "__main__":
    main()
