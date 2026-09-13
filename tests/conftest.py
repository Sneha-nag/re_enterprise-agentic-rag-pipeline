from pathlib import Path

import pytest

from meck_agent.embeddings import HashEmbeddings
from meck_agent.ingest_gis import load_sample_csvs
from meck_agent.rag import ingest_docs

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


@pytest.fixture
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "meck.db"
    load_sample_csvs(db_path, SAMPLE_DIR)
    return db_path


@pytest.fixture
def hash_embeddings() -> HashEmbeddings:
    return HashEmbeddings()


@pytest.fixture
def sample_chroma(tmp_path: Path, hash_embeddings: HashEmbeddings) -> Path:
    chroma_path = tmp_path / "chroma"
    ingest_docs(
        SAMPLE_DIR / "docs",
        chroma_path,
        include_remote=False,
        embeddings=hash_embeddings,
    )
    return chroma_path
