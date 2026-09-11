# Mecklenburg County RAG + Agentic CLI

Local learning project for a North Carolina broker: ask questions in the terminal about **Mecklenburg County open tax/sales data** and **NC Real Estate Commission / license-law notes**.

It is built to stay near **$0/month** (comfortably under $20) by using Groq or Gemini free tiers, **local embeddings**, SQLite, and Chroma on your machine.

This is a study tool, not legal, appraisal, or MLS software.

## What you learn

Two retrieval styles instead of dumping everything into a vector database:

| Kind of question | How it is answered |
| --- | --- |
| Dual agency, WWREA, license law | **RAG** over chunked guideline text in Chroma |
| PIN lookup, “what sold on Oak Street in 2024?” | **Agent tools** that query SQLite |

A LangGraph ReAct agent chooses the tool. County **recorded** `saleprice` / `saledate` values are **not MLS solds**.

```text
you (CLI) -> LangGraph agent -> search_guidelines (Chroma)
                              -> lookup_parcel / search_sales (SQLite)
                              -> Groq or Gemini
```

## Cost

| Piece | Default | Typical cost |
| --- | --- | --- |
| LLM | Groq (`llama-3.1-8b-instant`) or Gemini Flash | Free tier |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` on your CPU | $0 |
| Vector store | Local Chroma | $0 |
| Parcels / sales | Mecklenburg ArcGIS REST, attributes only | $0 |
| Optional backup | OpenAI `gpt-4o-mini` | Easy to keep under $20 |

Do not plug in paid MLS, Zillow, or hosted RAG platforms for this exercise.

## Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

Add **one** key to `.env`:

- Groq: https://console.groq.com/ (set `LLM_PROVIDER=groq` and `GROQ_API_KEY`)
- Gemini: https://aistudio.google.com/apikey (set `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY`). Gemini 3.x models (for example `gemini-3.6-flash`) require thinking / thought signatures on tool calls; the CLI enables that automatically.

If Hugging Face model download is slow, set `EMBEDDING_BACKEND=hash` in `.env`. Quality drops; ingest and tests still run.

## Load data

Fixture data (no network GIS). Enough to try the example questions:

```bash
meck-agent ingest --sample
meck-agent ingest-docs --sample-only
```

Live Mecklenburg open data (default: mailing ZIP prefix `28202`, max 500 parcels, **no polygons**):

```bash
meck-agent ingest --zip 28202 --limit 500
meck-agent ingest --street TRYON --city CHARLOTTE --limit 300
meck-agent ingest-docs
```

`ingest-docs` always indexes `data/sample/docs/` and, unless `--sample-only`, also tries to download [NC GS Chapter 93A](https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/ByChapter/Chapter_93A.html).

Show paths:

```bash
meck-agent paths
```

## Ask questions

```bash
meck-agent ask "What does NC require when I work with a buyer as a dual agent?"
meck-agent ask "Look up parcel 12501234 and its last recorded sale."
meck-agent ask "What sold on Oak Street in 2024 according to county records?"
meck-agent ask "Look up parcel 12501234" --verbose
meck-agent chat
```

`--verbose` prints tool calls so you can see the agent pick RAG vs SQL.

## Open data sources

- [Tax parcels with CAMA](https://meckgis.mecklenburgcountync.gov/server/rest/services/TaxParcel_camadata/MapServer) — PIN, address, land use, acres, assessed values, last CAMA sale
- [Tax Parcel Sales](https://meckgis.mecklenburgcountync.gov/server/rest/services/TaxParcelSales/FeatureServer) — recorded sales, grantor/grantee, `salesvalidity`
- [NCREC](https://www.ncrec.gov/) and [NC GS Chapter 93A](https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/ByChapter/Chapter_93A.html)

CAMA `zipcode` is often the **mailing** ZIP. Street and city filters are usually better for situs. Validity flags matter; related-party and quitclaim deeds still appear in the sales layer.

## Project layout

```text
src/meck_agent/     CLI, LangGraph agent, tools, ingest, RAG
data/sample/        Tiny CSVs + study notes so the CLI works offline
data/processed/     SQLite (gitignored)
data/chroma/        Vector index (gitignored)
tests/              Offline tests (hash embeddings, no paid APIs)
```

## Tests

```bash
pytest
```

Tests use bundled fixtures and hash embeddings. They do not call Groq, Gemini, or county GIS.

## Out of scope

Paid MLS, POLARIS scraping, maps, auth, hosting, full-county geometry, and anything you would ship to clients as advice.
