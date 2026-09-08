#!/usr/bin/env python3
"""Build the dense retrieval index over the tax regulation corpus.

Required before the retrieval settings (standard / summary / interact /
inlinesearch) can run.

    python scripts/build_index.py
    python scripts/build_index.py --encoder text-embedding-3-large
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxreasoning import data, retrieval  # noqa: E402

logger = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regulations", type=Path, default=data.DEFAULT_REGULATIONS)
    parser.add_argument("--encoder", default=retrieval.DEFAULT_ENCODER,
                        help=f"Embedding model (default: {retrieval.DEFAULT_ENCODER})")
    parser.add_argument("--backend", choices=["sentence-transformers", "openai"],
                        help="Override the backend inferred from the encoder name")
    parser.add_argument("--out", type=Path, default=retrieval.DEFAULT_INDEX)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    corpus = data.load_regulations(args.regulations)
    passages = retrieval.passages_from_regulations(corpus)
    logger.info("indexing %d regulation documents from %s", len(passages), args.regulations)

    encoder = retrieval.Encoder(args.encoder, args.backend)
    index = retrieval.Retriever.build(passages, encoder)
    index.save(args.out)


if __name__ == "__main__":
    main()
