"""Chroma retrieval layer used by the `knowledge_search` tool.

Embeddings: we use Chroma's built-in ONNX `all-MiniLM-L6-v2` (the
`DefaultEmbeddingFunction`) wrapped in LangChain's `Embeddings` interface.

Tradeoff: this model runs locally, costs nothing, and needs no torch install
(ONNX runtime only), so the demo works without extra API spend. It is less
accurate than Gemini embeddings on niche vocabulary. If answer quality matters
more than cost, swap `LocalMiniLMEmbeddings` for
`langchain_google_genai.GoogleGenerativeAIEmbeddings(model="models/embedding-001")`
and re-run ingest.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


class LocalMiniLMEmbeddings(Embeddings):
    """LangChain-compatible wrapper around Chroma's default ONNX embedder."""

    def __init__(self) -> None:
        # Imported lazily so importing this module never triggers a model download.
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._embed = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in self._embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._embed([text])[0]))


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """One shared embedding function (loading the model is the expensive part)."""
    return LocalMiniLMEmbeddings()


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Persistent Chroma collection. Creates the directory on first use."""
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def search_knowledge(query: str, k: int | None = None) -> str:
    """Return the top-k knowledge base chunks as readable, sourced text.

    Never raises: on failure it returns a message the agent can speak, so a
    vector-store problem does not kill the WebSocket.
    """
    settings = get_settings()
    k = k or settings.rag_top_k
    query = (query or "").strip()
    if not query:
        return "No search query was provided."

    try:
        docs = get_vectorstore().similarity_search(query, k=k)
    except Exception:
        logger.exception("Knowledge base search failed for query=%r", query)
        return "Sorry, I'm having trouble looking that up right now."

    if not docs:
        return (
            "No matching information was found in the knowledge base. "
            "Tell the lead honestly that you will have a human follow up with the details."
        )

    chunks = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "knowledge_base")
        chunks.append(f"[{index}] ({source})\n{doc.page_content.strip()}")
    return "\n\n".join(chunks)
