from __future__ import annotations

import typer
from dotenv import load_dotenv

from meck_agent.config import Settings, get_settings

load_dotenv()

app = typer.Typer(
    add_completion=False,
    help="Mecklenburg County learning RAG agent (open tax data + NC guidelines).",
)


@app.callback()
def _root() -> None:
    """Mecklenburg County real-estate Q&A CLI."""


@app.command()
def ingest(
    sample: bool = typer.Option(
        False, "--sample", help="Load bundled fixture CSVs instead of calling county GIS."
    ),
    zipcode: str = typer.Option("28202", "--zip", help="CAMA mailing ZIP prefix (live ingest)."),
    street: str = typer.Option("", "--street", help="Optional street name fragment."),
    city: str = typer.Option("", "--city", help="Optional location city fragment."),
    limit: int = typer.Option(500, "--limit", help="Max CAMA parcels to download."),
) -> None:
    """Load parcels and recorded sales into local SQLite (no polygon geometry)."""
    from meck_agent.ingest_gis import ingest_gis_live, load_sample_csvs

    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    if sample:
        counts = load_sample_csvs(settings.db_path, settings.sample_dir)
        typer.echo(
            f"Loaded sample fixtures into {settings.db_path}: "
            f"{counts['parcels']} parcels, {counts['sales']} sales."
        )
        return
    typer.echo(
        f"Downloading Mecklenburg CAMA subset zip={zipcode!r} street={street!r} "
        f"city={city!r} limit={limit} (attributes only)..."
    )
    counts = ingest_gis_live(
        settings.db_path,
        zipcode=zipcode or None,
        street=street or None,
        city=city or None,
        limit=limit,
    )
    typer.echo(
        f"Wrote {counts['parcels']} parcels and {counts['sales']} sales to {settings.db_path}"
    )


@app.command("ingest-docs")
def ingest_docs_cmd(
    sample_only: bool = typer.Option(
        False,
        "--sample-only",
        help="Index bundled study notes only (skip ncleg.gov download).",
    ),
) -> None:
    """Chunk and embed NC guideline text into local Chroma."""
    from meck_agent.rag import ingest_docs

    settings = get_settings()
    docs_dir = settings.sample_dir / "docs"
    typer.echo(f"Indexing documents from {docs_dir} -> {settings.chroma_path}")
    counts = ingest_docs(
        docs_dir,
        settings.chroma_path,
        include_remote=not sample_only,
        settings=settings,
    )
    typer.echo(
        f"Indexed {counts['source_docs']} sources "
        f"({counts['remote_docs']} remote) into {counts['chunks']} chunks."
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to send to the agent."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print tool calls."),
) -> None:
    """Ask one question (requires GROQ_API_KEY or GOOGLE_API_KEY)."""
    from meck_agent.agent import ask_question

    settings = get_settings()
    answer = ask_question(question, settings, verbose=verbose)
    typer.echo(answer)


@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print tool calls."),
) -> None:
    """Interactive loop. Type quit to exit."""
    from meck_agent.agent import ask_question

    settings = get_settings()
    typer.echo("Mecklenburg RAG agent. Type quit to exit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        try:
            answer = ask_question(question, settings, verbose=verbose)
            typer.echo(answer)
        except Exception as exc:
            typer.echo(f"Error: {exc}")


@app.command()
def paths() -> None:
    """Show resolved data locations."""
    settings: Settings = get_settings()
    typer.echo(f"root:    {settings.root}")
    typer.echo(f"db:      {settings.db_path} exists={settings.db_path.exists()}")
    typer.echo(f"chroma:  {settings.chroma_path}")
    typer.echo(f"sample:  {settings.sample_dir}")
    typer.echo(f"llm:     {settings.llm_provider}")


if __name__ == "__main__":
    app()
