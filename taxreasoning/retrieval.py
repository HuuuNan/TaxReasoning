"""Dense retrieval over the tax regulation corpus.

The paper's retrieval-augmented settings ("the model receives the top-5 most
relevant tax documents retrieved via a dense retriever") need an index over the
regulation corpus. The AAAI camera-ready puts the retriever's identity in an
appendix that is not part of the published PDF, so the encoder is configurable
here; ``bge-m3`` is the default because it is the usual choice for Chinese
legal/financial retrieval and handles the corpus's long documents.

Two backends:

``sentence-transformers``
    Local model, default ``BAAI/bge-m3``. Needs a GPU to be quick.
``openai``
    ``text-embedding-3-large`` through the API. No local model needed.

Set the encoder with ``--encoder`` on ``scripts/build_index.py``; the choice is
recorded in the index file so retrieval cannot silently mix encoders.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "BAAI/bge-m3"
DEFAULT_INDEX = Path("index/regulations.npz")


@dataclass
class Passage:
    """One retrievable unit: a regulation document."""

    doc_id: str
    title: str
    contents: str

    def to_dict(self) -> dict:
        return {"id": self.doc_id, "title": self.title, "contents": self.contents}


class Encoder:
    """Wraps whichever embedding backend the index was built with."""

    def __init__(self, name: str = DEFAULT_ENCODER, backend: str | None = None):
        self.name = name
        self.backend = backend or ("openai" if name.startswith("text-embedding") else "sentence-transformers")
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            logger.info("loading encoder %s (this can take a minute)", self.name)
            self._model = SentenceTransformer(self.name)
        elif self.backend == "openai":
            from taxreasoning import llm

            self._model = llm.make_client(self.name, provider="openai")
        else:
            raise ValueError(f"unknown encoder backend {self.backend!r}")

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed texts and L2-normalise, so dot product is cosine similarity."""
        self._load()

        if self.backend == "sentence-transformers":
            vectors = self._model.encode(
                texts, batch_size=batch_size, show_progress_bar=len(texts) > 1, normalize_embeddings=True
            )
            return np.asarray(vectors, dtype=np.float32)

        vectors = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            response = self._model.embeddings.create(model=self.name, input=chunk)
            vectors.extend(item.embedding for item in response.data)
        array = np.asarray(vectors, dtype=np.float32)
        return array / np.linalg.norm(array, axis=1, keepdims=True)


class Retriever:
    """A dense index over passages, searched by cosine similarity."""

    def __init__(self, passages: list[Passage], embeddings: np.ndarray, encoder: Encoder):
        self.passages = passages
        self.embeddings = embeddings
        self.encoder = encoder

    @classmethod
    def build(cls, passages: list[Passage], encoder: Encoder) -> "Retriever":
        texts = [f"{p.title}\n{p.contents}" for p in passages]
        logger.info("encoding %d passages with %s", len(texts), encoder.name)
        return cls(passages, encoder.encode(texts), encoder)

    def save(self, path: Path | str = DEFAULT_INDEX) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            passages=json.dumps([p.to_dict() for p in self.passages], ensure_ascii=False),
            encoder=self.encoder.name,
            backend=self.encoder.backend,
        )
        logger.info("wrote %s (%d passages, dim %d)", path, len(self.passages), self.embeddings.shape[1])

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX) -> "Retriever":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no index at {path}. Build one first:\n"
                f"    python scripts/build_index.py"
            )
        data = np.load(path, allow_pickle=False)
        raw = json.loads(str(data["passages"]))
        passages = [Passage(p["id"], p["title"], p["contents"]) for p in raw]
        encoder = Encoder(str(data["encoder"]), str(data["backend"]))
        return cls(passages, data["embeddings"], encoder)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the ``top_k`` passages most similar to ``query``.

        Result dicts use ALCE's field names (``id`` / ``title`` / ``contents``)
        so they drop straight into the prompt builders and the retrieval dumps.
        """
        query_vec = self.encoder.encode([query])[0]
        scores = self.embeddings @ query_vec
        order = np.argsort(-scores)[:top_k]
        results = []
        for rank, idx in enumerate(order):
            passage = self.passages[int(idx)]
            results.append({**passage.to_dict(), "score": float(scores[idx]), "rank": rank})
        return results


def passages_from_regulations(corpus: dict[str, str]) -> list[Passage]:
    """Turn the ``{title: text}`` regulation corpus into retrievable passages."""
    return [
        Passage(doc_id=str(i), title=title, contents=str(content))
        for i, (title, content) in enumerate(corpus.items())
    ]
