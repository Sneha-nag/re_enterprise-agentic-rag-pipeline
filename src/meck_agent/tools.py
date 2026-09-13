from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from meck_agent.db import compact_id, connect
from meck_agent.rag import retrieve


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(row) for row in rows]


def lookup_parcel_rows(db_path: Path, pin_or_address: str, limit: int = 8) -> list[dict]:
    query = (pin_or_address or "").strip()
    if not query:
        return []
    conn = connect(db_path)
    try:
        compact = compact_id(query)
        looks_like_id = (
            " " not in query
            and any(ch.isdigit() for ch in compact)
            and len(compact) >= 6
        )
        if looks_like_id:
            rows = conn.execute(
                """
                SELECT * FROM parcels
                WHERE replace(replace(parcel_id, '-', ''), ' ', '') = ?
                   OR replace(replace(IFNULL(nc_pin, ''), '-', ''), ' ', '') = ?
                LIMIT ?
                """,
                (compact, compact, limit),
            ).fetchall()
            if rows:
                return _rows_to_dicts(rows)
        like = f"%{query.upper()}%"
        rows = conn.execute(
            """
            SELECT * FROM parcels
            WHERE UPPER(IFNULL(address, '')) LIKE ?
               OR UPPER(IFNULL(street_name, '')) LIKE ?
               OR (IFNULL(street_number, '') || ' ' || IFNULL(street_name, '')) LIKE ?
            LIMIT ?
            """,
            (like, like, like, limit),
        ).fetchall()
        return _rows_to_dicts(rows)
    finally:
        conn.close()


def search_sales_rows(
    db_path: Path,
    *,
    street: str | None = None,
    zipcode: str | None = None,
    year: int | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    valid_only: bool = False,
    limit: int = 15,
) -> list[dict]:
    clauses = ["1=1"]
    params: list[object] = []
    if street:
        clauses.append("UPPER(IFNULL(p.street_name, '')) LIKE ?")
        params.append(f"%{street.strip().upper()}%")
    if zipcode:
        clauses.append("IFNULL(p.zipcode, '') LIKE ?")
        params.append(f"{zipcode.strip()}%")
    if year:
        clauses.append("substr(IFNULL(s.sale_date, ''), 1, 4) = ?")
        params.append(str(year))
    if min_price is not None:
        clauses.append("s.sale_price >= ?")
        params.append(min_price)
    if max_price is not None:
        clauses.append("s.sale_price <= ?")
        params.append(max_price)
    if valid_only:
        clauses.append("UPPER(IFNULL(s.sales_validity, '')) IN ('Y', 'YES', 'V', 'VALID')")
    params.append(limit)
    sql = f"""
        SELECT
            s.parcel_id,
            s.sale_price,
            s.sale_date,
            s.sales_validity,
            s.sold_as_vacant,
            s.land_use_description,
            s.grantor,
            s.grantee,
            s.legal_reference,
            p.address,
            p.street_name,
            p.city,
            p.zipcode,
            p.heated_area,
            p.bedrooms,
            p.year_built
        FROM sales s
        LEFT JOIN parcels p ON p.parcel_id = s.parcel_id
        WHERE {' AND '.join(clauses)}
        ORDER BY s.sale_date DESC
        LIMIT ?
    """
    conn = connect(db_path)
    try:
        return _rows_to_dicts(conn.execute(sql, params).fetchall())
    finally:
        conn.close()


def search_guideline_hits(
    chroma_path: Path,
    query: str,
    *,
    k: int = 5,
    embeddings=None,
) -> list[dict]:
    docs = retrieve(chroma_path, query, k=k, embeddings=embeddings)
    hits = []
    for doc in docs:
        meta = doc.metadata or {}
        hits.append(
            {
                "title": meta.get("title") or meta.get("source"),
                "source": meta.get("source"),
                "url": meta.get("url") or "",
                "excerpt": doc.page_content.strip()[:1200],
            }
        )
    return hits


def make_tools(db_path: Path, chroma_path: Path, embeddings=None):
    """LangChain tools bound to local SQLite + Chroma paths."""

    @tool
    def search_guidelines(query: str) -> str:
        """Search NC Real Estate Commission / license-law notes and related public guidelines.

        Use for dual agency, WWREA disclosure, licensing, trust money, and similar
        practice questions. Returns excerpts with source titles.
        """
        hits = search_guideline_hits(
            chroma_path, query, k=5, embeddings=embeddings
        )
        if not hits:
            return json.dumps(
                {
                    "error": "No guideline chunks indexed. Run: meck-agent ingest-docs",
                    "results": [],
                }
            )
        return json.dumps({"results": hits}, indent=2)

    @tool
    def lookup_parcel(pin_or_address: str) -> str:
        """Look up a Mecklenburg tax parcel by PIN / parcel id or street address.

        Returns assessed values, land use, heated area, and last recorded CAMA sale.
        """
        if not db_path.exists():
            return json.dumps(
                {"error": "SQLite DB missing. Run: meck-agent ingest --sample", "results": []}
            )
        rows = lookup_parcel_rows(db_path, pin_or_address)
        if not rows:
            return json.dumps(
                {
                    "results": [],
                    "note": "No parcel matched. Try a PIN or a street fragment. "
                    "The local DB may only contain a ZIP/street subset.",
                }
            )
        return json.dumps({"results": rows}, indent=2, default=str)

    @tool
    def search_sales(
        street: str = "",
        zipcode: str = "",
        year: int = 0,
        min_price: float = 0,
        max_price: float = 0,
        valid_only: bool = False,
        limit: int = 15,
    ) -> str:
        """Search recorded (tax/assessor) sales, not MLS listings.

        Filter by street name fragment, ZIP, year, and optional price band.
        Always remind the user these are county recorded sales, not MLS comps.
        """
        if not db_path.exists():
            return json.dumps(
                {"error": "SQLite DB missing. Run: meck-agent ingest --sample", "results": []}
            )
        year_val = int(year) if year else None
        min_val = float(min_price) if min_price else None
        max_val = float(max_price) if max_price else None
        limit_val = int(limit) if limit else 15
        rows = search_sales_rows(
            db_path,
            street=street or None,
            zipcode=zipcode or None,
            year=year_val,
            min_price=min_val,
            max_price=max_val,
            valid_only=bool(valid_only),
            limit=min(limit_val, 25),
        )
        return json.dumps(
            {
                "results": rows,
                "disclaimer": (
                    "These are Mecklenburg County recorded/tax sales, not MLS closed sales. "
                    "Check sales_validity before treating a price as market evidence."
                ),
            },
            indent=2,
            default=str,
        )

    return [search_guidelines, lookup_parcel, search_sales]
