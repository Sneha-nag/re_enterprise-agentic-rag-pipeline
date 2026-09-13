# Mecklenburg County RAG Agent (Streamlit)

Local learning project for a North Carolina broker: a **browser chat UI** that answers questions about **Mecklenburg County open tax/sales data** and **NC Real Estate Commission / license-law notes**.

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
you (Streamlit) -> LangGraph agent -> search_guidelines (Chroma)
                                   -> lookup_parcel / search_sales (SQLite)
                                   -> Groq or Gemini
```

## Cost

| Piece | Default | Typical cost |
| --- | --- | --- |
| LLM | Groq (`llama-3.1-8b-instant`) or Gemini 3.6 Flash | Free tier |
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
- Gemini: https://aistudio.google.com/apikey (set `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY`). Gemini 3.x models (for example `gemini-3.6-flash`) require thinking / thought signatures on tool calls; the app enables that automatically.

If Hugging Face model download is slow, set `EMBEDDING_BACKEND=hash` in `.env`. Quality drops; ingest and tests still run.

## Run the UI

```bash
streamlit run app.py
```

or:

```bash
meck-agent ui
```

Then open the URL Streamlit prints (usually http://localhost:8501).

In the sidebar:

1. **Load sample parcels & sales**
2. **Index sample guidelines**
3. Ask a question, or click an example

Live county download (ZIP/street subset, no polygons) is also in the sidebar.

## Example questions

- What does NC require when I work with a buyer as a dual agent?
- Look up parcel 12501234 and its last recorded sale.
- What sold on Oak Street in 2024 according to county records?

Optional terminal commands still work for ingest and one-off asks:

```bash
meck-agent ingest --sample
meck-agent ingest-docs --sample-only
meck-agent ask "Look up parcel 12501234"
```

## Open data sources

- [Tax parcels with CAMA](https://meckgis.mecklenburgcountync.gov/server/rest/services/TaxParcel_camadata/MapServer) — PIN, address, land use, acres, assessed values, last CAMA sale
- [Tax Parcel Sales](https://meckgis.mecklenburgcountync.gov/server/rest/services/TaxParcelSales/FeatureServer) — recorded sales, grantor/grantee, `salesvalidity`
- [NCREC](https://www.ncrec.gov/) and [NC GS Chapter 93A](https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/ByChapter/Chapter_93A.html)

CAMA `zipcode` is often the **mailing** ZIP. Street and city filters are usually better for situs. Validity flags matter; related-party and quitclaim deeds still appear in the sales layer.

## Project layout

```text
app.py              Streamlit entry point
src/meck_agent/     UI, LangGraph agent, tools, ingest, RAG
data/sample/        Tiny CSVs + study notes so the app works offline
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
