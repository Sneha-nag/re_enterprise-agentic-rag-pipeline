from __future__ import annotations

from meck_agent.config import Settings
from meck_agent.db import connect


def _table_count(db_path, table: str) -> int:
    if not db_path.exists():
        return 0
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not row or row[0] == 0:
            return 0
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(count[0]) if count else 0
    finally:
        conn.close()


def provider_model_name(settings: Settings) -> str:
    provider = (settings.llm_provider or "groq").strip().lower()
    if provider in {"gemini", "google"}:
        return settings.gemini_model
    if provider == "openai":
        return settings.openai_model
    return settings.groq_model


def has_llm_key(settings: Settings) -> bool:
    provider = (settings.llm_provider or "groq").strip().lower()
    if provider in {"gemini", "google"}:
        return bool(settings.google_api_key)
    if provider == "openai":
        return bool(settings.openai_api_key)
    return bool(settings.groq_api_key)


def chroma_ready(chroma_path) -> bool:
    if not chroma_path.exists():
        return False
    return any(chroma_path.iterdir())


def data_status(settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    return {
        "provider": (settings.llm_provider or "groq").strip().lower(),
        "model": provider_model_name(settings),
        "has_api_key": has_llm_key(settings),
        "db_path": str(settings.db_path),
        "db_exists": settings.db_path.exists(),
        "parcel_count": _table_count(settings.db_path, "parcels"),
        "sales_count": _table_count(settings.db_path, "sales"),
        "chroma_path": str(settings.chroma_path),
        "chroma_ready": chroma_ready(settings.chroma_path),
        "sample_dir": str(settings.sample_dir),
    }
