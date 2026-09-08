# TaxReasoning

Code and data for **TaxReasoning: Benchmarking Knowledge-Intensive Mathematical Reasoning with Evolving Tax Laws** (AAAI-26).


The benchmark pairs tax computation problems with the specific
regulations needed to solve them. Because those regulations are amended over
time, a model must apply the version in force rather than a memorised rate.

This repository currently holds the evaluation code only.
Benchmark data is distributed separately, see [data/README.md](data/README.md).

## Setup

```bash
pip install -r requirements.txt
```


## Methods

| `--mode` | Paper | Context given to the model |
|---|---|---|
| `cot` | Vanilla CoT | Nothing — closed-book chain of thought |
| `pot` | Vanilla PoT | Nothing; the model writes a `solution()` function |
| `standard` | Single-turn RAG | Top-5 documents from the dense retriever |
| `summary` | Single-turn RAG | Those documents summarised by GPT-4o and relevance-filtered |
| `interact` | Multi-turn RAG | Summaries of the top-10; model issues `Check: Document [i]` / `Output:` / `End.` |
| `inlinesearch` | Multi-turn RAG | Model issues `Search: query` / `Output:` / `End.` mid-reasoning |
| `oracle` | Oracle | The gold regulations annotated with each question |

The two multi-turn protocols follow ALCE.

## Running

```bash
# Vanilla — no retrieval, no index needed
python scripts/run_predict.py --model gpt-4o-2024-08-06 --mode cot
python scripts/run_predict.py --model gpt-4o-2024-08-06 --mode pot

# Oracle
python scripts/run_predict.py --model deepseek-chat --mode oracle

# Retrieval settings — build the index once first
python scripts/build_index.py
python scripts/run_predict.py --model deepseek-chat --mode standard
python scripts/run_predict.py --model deepseek-chat --mode summary
python scripts/run_predict.py --model deepseek-chat --mode interact
python scripts/run_predict.py --model deepseek-chat --mode inlinesearch
```

Provider routing is inferred from the model name (`gpt-*`/`o*` → OpenAI,
`deepseek-*` → DeepSeek, everything else → the aggregator); override with
`--provider`.

Predictions are written as JSONL to `results/<model>-<mode>.jsonl`. Re-running
the same command resumes and skips questions already in the file.

## Scoring

GPT-4o-mini extracts the final numerical answer from the reasoning trace (or,
for `pot`, the generated `solution()` is executed), then the number is compared
to the gold answer with a ±0.2% tolerance.

```bash
python scripts/run_eval.py --pred results/deepseek-chat-oracle.jsonl
python scripts/run_eval.py --pred results/deepseek-chat-pot.jsonl --allow-exec
```

Writes a per-example `*-eval.jsonl` plus a `*-eval.summary.json` with the accuracy.

> `pot` scoring executes Python written by the evaluated model in the current
> process. There is a 10 s timeout, but no sandbox.

## Other scripts

```bash
# external calculator: re-evaluate the model's own expression with SymPy
python scripts/run_calculator.py --pred results/deepseek-chat-cot.jsonl

# win-tie-lose between two scored runs
python scripts/analyze_wtl.py --a A-eval.jsonl --b B-eval.jsonl

# check that every regulation cited by a question is in the corpus
python tools/check_data.py
```

`scripts/eval_citation.py` holds the ALCE citation metrics.

## Attribution

The citation metrics and the two multi-turn retrieval protocols derive from
[ALCE](https://github.com/princeton-nlp/ALCE) (MIT, © 2023 Princeton NLP). See
[NOTICE](NOTICE); each derived file records what was changed relative to
upstream.
