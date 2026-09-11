from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Prefer MECK_HOME, else walk up from cwd looking for project markers."""
    import os

    env_home = os.getenv("MECK_HOME")
    if env_home:
        return Path(env_home).resolve()

    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data" / "sample").exists():
            return candidate
        if (candidate / "data" / "sample" / "parcels.csv").exists():
            return candidate
    return here


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_backend: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    meck_home: Path | None = None
    default_zip: str = "28202"

    @property
    def root(self) -> Path:
        if self.meck_home:
            return Path(self.meck_home).resolve()
        return find_project_root()

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "processed" / "meck.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"


def get_settings() -> Settings:
    return Settings()
