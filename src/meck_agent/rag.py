from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from meck_agent.config import Settings
from meck_agent.embeddings import get_embeddings

COLLECTION_NAME = "meck_guidelines"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

REMOTE_DOCS = (
    {
        "title": "NC GS Chapter 93A Real Estate License Law",
        "source": "ncleg.gov",
        "url": "https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/ByChapter/Chapter_93A.html",
    },
)


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def load_local_docs(sample_docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    if not sample_docs_dir.exists():
        return documents
    for path in sorted(sample_docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".html"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "title": path.stem.replace("_", " "),
                    "source": str(path.name),
                    "url": "",
                },
            )
        )
    return documents


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def pdf_to_text(data: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def fetch_remote_docs(timeout: float = 45.0) -> list[Document]:
    import httpx

    documents: list[Document] = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for spec in REMOTE_DOCS:
            try:
                response = client.get(spec["url"])
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "pdf" in content_type or spec["url"].lower().endswith(".pdf"):
                    text = pdf_to_text(response.content)
                else:
                    text = html_to_text(response.text)
                text = text.strip()
                if len(text) < 200:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "title": spec["title"],
                            "source": spec["source"],
                            "url": spec["url"],
                        },
                    )
                )
            except Exception as exc:
                print(f"Skipping {spec['url']}: {exc}")
    return documents


def get_vectorstore(chroma_path: Path, embeddings=None, settings: Settings | None = None):
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    from langchain_chroma import Chroma

    embeddings = embeddings or get_embeddings(settings)
    chroma_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )


def _reset_collection(chroma_path: Path) -> None:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def index_documents(
    chroma_path: Path,
    documents: list[Document],
    *,
    embeddings=None,
    settings: Settings | None = None,
    replace: bool = True,
) -> int:
    chunks = _split_documents(documents)
    if not chunks:
        return 0
    if replace:
        _reset_collection(chroma_path)
    store = get_vectorstore(chroma_path, embeddings=embeddings, settings=settings)
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def retrieve(
    chroma_path: Path,
    query: str,
    *,
    k: int = 5,
    embeddings=None,
    settings: Settings | None = None,
) -> list[Document]:
    store = get_vectorstore(chroma_path, embeddings=embeddings, settings=settings)
    return store.similarity_search(query, k=k)


def ingest_docs(
    sample_docs_dir: Path,
    chroma_path: Path,
    *,
    include_remote: bool = True,
    embeddings=None,
    settings: Settings | None = None,
) -> dict[str, int]:
    documents = load_local_docs(sample_docs_dir)
    remote = fetch_remote_docs() if include_remote else []
    documents.extend(remote)
    n_chunks = index_documents(
        chroma_path,
        documents,
        embeddings=embeddings,
        settings=settings,
        replace=True,
    )
    return {
        "source_docs": len(documents),
        "chunks": n_chunks,
        "remote_docs": len(remote),
    }
