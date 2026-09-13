from __future__ import annotations

from langchain_core.embeddings import Embeddings

from meck_agent.config import Settings


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-token hashing. Used in tests and as a no-download fallback."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        import hashlib
        import struct

        vec = [0.0] * self.dim
        tokens = [tok for tok in text.lower().replace("\n", " ").split() if tok]
        if not tokens:
            tokens = ["empty"]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(0, 32, 4):
                idx = struct.unpack_from(">I", digest, i)[0] % self.dim
                vec[idx] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or Settings()
    backend = (settings.embedding_backend or "huggingface").lower()
    if backend == "hash":
        return HashEmbeddings()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    except Exception as exc:  # pragma: no cover - environment-dependent
        import warnings

        warnings.warn(
            f"Falling back to hash embeddings ({exc}). "
            "Install sentence-transformers for better RAG quality.",
            stacklevel=2,
        )
        return HashEmbeddings()
