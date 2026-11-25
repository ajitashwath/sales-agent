"""Ingest the sample knowledge base into Chroma.

Usage (from the project root):

    python -m app.rag.ingest

Reads every *.txt/.md/*.pdf in data/knowledge_base/, chunks it with
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) and rebuilds
the Chroma collection from scratch so re-running is idempotent.

Embeddings are local + free (see app/rag/retriever.py for the tradeoff note).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python app/rag/ingest.py` as well as `python -m app.rag.ingest`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.retriever import get_embeddings  # noqa: E402

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_documents(directory: Path) -> list[Document]:
    """Load every supported file in `directory` as a LangChain Document."""
    documents: list[Document] = []

    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.lower() == ".pdf":
            # PDF support is optional; pypdf is in requirements.txt.
            from pypdf import PdfReader

            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": path.name}))
            print(f"  loaded {path.name} ({len(text)} chars)")

    return documents


def ingest(reset: bool = True) -> int:
    """Chunk + embed + persist. Returns the number of chunks written."""
    settings = get_settings()
    kb_dir = settings.knowledge_base_dir

    if not kb_dir.exists():
        raise SystemExit(f"Knowledge base directory not found: {kb_dir}")

    print(f"Reading documents from {kb_dir}")
    documents = load_documents(kb_dir)
    if not documents:
        raise SystemExit(
            f"No .txt/.md/.pdf documents found in {kb_dir}. Add a few docs and retry."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s)")

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
            print(f"Deleted existing collection '{settings.chroma_collection}'")
        except Exception:
            # Collection did not exist yet - nothing to delete.
            pass

    vectorstore = Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
    )

    # Deterministic ids so a failed ingest can simply be re-run.
    ids = [f"{doc.metadata.get('source', 'doc')}-{i}" for i, doc in enumerate(chunks)]
    vectorstore.add_documents(chunks, ids=ids)

    print(
        f"Stored {len(chunks)} chunk(s) in Chroma collection "
        f"'{settings.chroma_collection}' at {settings.chroma_dir}"
    )
    return len(chunks)


if __name__ == "__main__":
    ingest()
