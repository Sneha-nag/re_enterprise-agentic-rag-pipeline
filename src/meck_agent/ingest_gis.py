from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import httpx

from meck_agent.db import PARCEL_FIELDS, SALE_FIELDS, connect, epoch_to_iso_date, init_db

CAMA_URL = (
    "https://meckgis.mecklenburgcountync.gov/server/rest/services/"
    "TaxParcel_camadata/MapServer/0/query"
)
SALES_URL = (
    "https://meckgis.mecklenburgcountync.gov/server/rest/services/"
    "TaxParcelSales/FeatureServer/0/query"
)

CAMA_OUT_FIELDS = (
    "parcelid,nc_pin,pid,address,streetnumber,streetname,loccity,city,zipcode,"
    "neighborhood,lusecode,landuse_description,totalac,heatedarea,bedrooms,"
    "yearbuilt,totalvalue,totmarkval,saleprice,saledate,validsale,"
    "ownrlstnme,ownrfrstnme"
)
SALES_OUT_FIELDS = (
    "parcelid,saleprice,saledate,salesvalidity,soldasvacantflag,landuse,"
    "landusefulldescription,grantor,grantee,deeddescription,legalreference"
)

PAGE_SIZE = 1000
SALES_ID_BATCH = 40


def _get(attrs: dict, *keys: str, default=None):
    for key in keys:
        if key in attrs and attrs[key] not in (None, ""):
            return attrs[key]
        lower = {k.lower(): v for k, v in attrs.items()}
        if key.lower() in lower and lower[key.lower()] not in (None, ""):
            return lower[key.lower()]
    return default


def cama_to_parcel(attrs: dict) -> dict:
    street_name = _get(attrs, "streetname")
    if isinstance(street_name, str):
        street_name = street_name.strip().upper()
    city = _get(attrs, "loccity", "city")
    if isinstance(city, str):
        city = city.strip().upper()
    return {
        "parcel_id": str(_get(attrs, "parcelid", "pid") or "").strip(),
        "nc_pin": _get(attrs, "nc_pin"),
        "address": _get(attrs, "address"),
        "street_number": _get(attrs, "streetnumber"),
        "street_name": street_name,
        "city": city,
        "zipcode": str(_get(attrs, "zipcode") or "").split("-")[0],
        "neighborhood": _get(attrs, "neighborhood"),
        "land_use": _get(attrs, "lusecode"),
        "land_use_description": _get(attrs, "landuse_description"),
        "acres": _get(attrs, "totalac"),
        "heated_area": _get(attrs, "heatedarea"),
        "bedrooms": _get(attrs, "bedrooms"),
        "year_built": _get(attrs, "yearbuilt"),
        "total_value": _get(attrs, "totalvalue"),
        "market_value": _get(attrs, "totmarkval"),
        "last_sale_price": _get(attrs, "saleprice"),
        "last_sale_date": epoch_to_iso_date(_get(attrs, "saledate")),
        "valid_sale": _get(attrs, "validsale"),
        "owner_last": _get(attrs, "ownrlstnme"),
        "owner_first": _get(attrs, "ownrfrstnme"),
    }


def sales_to_row(attrs: dict) -> dict:
    return {
        "parcel_id": str(_get(attrs, "parcelid") or "").strip(),
        "sale_price": _get(attrs, "saleprice"),
        "sale_date": epoch_to_iso_date(_get(attrs, "saledate")),
        "sales_validity": _get(attrs, "salesvalidity"),
        "sold_as_vacant": _get(attrs, "soldasvacantflag"),
        "land_use": _get(attrs, "landuse"),
        "land_use_description": _get(attrs, "landusefulldescription"),
        "grantor": _get(attrs, "grantor"),
        "grantee": _get(attrs, "grantee"),
        "deed_description": _get(attrs, "deeddescription"),
        "legal_reference": _get(attrs, "legalreference"),
    }


def _upsert_parcels(conn: sqlite3.Connection, rows: list[dict]) -> int:
    placeholders = ",".join("?" for _ in PARCEL_FIELDS)
    sql = (
        f"INSERT OR REPLACE INTO parcels ({', '.join(PARCEL_FIELDS)}) "
        f"VALUES ({placeholders})"
    )
    count = 0
    for row in rows:
        if not row.get("parcel_id"):
            continue
        conn.execute(sql, [row.get(field) for field in PARCEL_FIELDS])
        count += 1
    return count


def _insert_sales(conn: sqlite3.Connection, rows: list[dict], replace: bool) -> int:
    if replace:
        conn.execute("DELETE FROM sales")
    placeholders = ",".join("?" for _ in SALE_FIELDS)
    sql = f"INSERT INTO sales ({', '.join(SALE_FIELDS)}) VALUES ({placeholders})"
    count = 0
    for row in rows:
        if not row.get("parcel_id"):
            continue
        conn.execute(sql, [row.get(field) for field in SALE_FIELDS])
        count += 1
    return count


def load_sample_csvs(db_path: Path, sample_dir: Path, replace: bool = True) -> dict[str, int]:
    init_db(db_path)
    conn = connect(db_path)
    try:
        if replace:
            conn.execute("DELETE FROM parcels")
            conn.execute("DELETE FROM sales")
        parcels: list[dict] = []
        with (sample_dir / "parcels.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parcels.append({k: (v if v != "" else None) for k, v in row.items()})
        sales: list[dict] = []
        with (sample_dir / "sales.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sales.append({k: (v if v != "" else None) for k, v in row.items()})
        n_parcels = _upsert_parcels(conn, parcels)
        n_sales = _insert_sales(conn, sales, replace=False)
        conn.commit()
        return {"parcels": n_parcels, "sales": n_sales}
    finally:
        conn.close()


def build_cama_where(
    zipcode: str | None = None,
    street: str | None = None,
    city: str | None = None,
) -> str:
    clauses: list[str] = ["parcelid IS NOT NULL"]
    if zipcode:
        safe = zipcode.replace("'", "")
        clauses.append(f"zipcode LIKE '{safe}%'")
    if street:
        safe = street.replace("'", "").upper()
        clauses.append(f"UPPER(streetname) LIKE '%{safe}%'")
    if city:
        safe = city.replace("'", "").upper()
        clauses.append(f"UPPER(loccity) LIKE '%{safe}%'")
    return " AND ".join(clauses)


def query_arcgis(
    url: str,
    where: str,
    out_fields: str,
    *,
    max_records: int | None = None,
    page_size: int = PAGE_SIZE,
    client: httpx.Client | None = None,
) -> list[dict]:
    own_client = client is None
    client = client or httpx.Client(timeout=60.0)
    features: list[dict] = []
    offset = 0
    try:
        while True:
            remaining = page_size
            if max_records is not None:
                remaining = min(page_size, max_records - len(features))
                if remaining <= 0:
                    break
            payload = {
                "where": where,
                "outFields": out_fields,
                "returnGeometry": "false",
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": remaining,
            }
            response = client.post(url, data=payload)
            response.raise_for_status()
            body = response.json()
            if body.get("error"):
                raise RuntimeError(f"ArcGIS error: {body['error']}")
            batch = body.get("features") or []
            for feature in batch:
                attrs = feature.get("attributes") or {}
                features.append(attrs)
            if not batch:
                break
            offset += len(batch)
            if max_records is not None and len(features) >= max_records:
                break
            if not body.get("exceededTransferLimit"):
                break
        return features
    finally:
        if own_client:
            client.close()


def ingest_gis_live(
    db_path: Path,
    *,
    zipcode: str | None = "28202",
    street: str | None = None,
    city: str | None = None,
    limit: int = 500,
    client: httpx.Client | None = None,
) -> dict[str, int]:
    """Download a CAMA subset (no geometry) and matching TaxParcelSales rows."""
    init_db(db_path)
    where = build_cama_where(zipcode=zipcode, street=street, city=city)
    cama_rows = query_arcgis(
        CAMA_URL,
        where,
        CAMA_OUT_FIELDS,
        max_records=limit,
        client=client,
    )
    parcels = [cama_to_parcel(attrs) for attrs in cama_rows]
    parcels = [p for p in parcels if p.get("parcel_id")]

    parcel_ids = [p["parcel_id"] for p in parcels]
    sales_rows: list[dict] = []
    own_client = client is None
    client = client or httpx.Client(timeout=60.0)
    try:
        for i in range(0, len(parcel_ids), SALES_ID_BATCH):
            batch = parcel_ids[i : i + SALES_ID_BATCH]
            quoted = ",".join("'" + pid.replace("'", "") + "'" for pid in batch)
            sales_where = f"parcelid IN ({quoted})"
            raw = query_arcgis(
                SALES_URL,
                sales_where,
                SALES_OUT_FIELDS,
                client=client,
            )
            sales_rows.extend(sales_to_row(attrs) for attrs in raw)
    finally:
        if own_client:
            client.close()

    conn = connect(db_path)
    try:
        conn.execute("DELETE FROM parcels")
        conn.execute("DELETE FROM sales")
        n_parcels = _upsert_parcels(conn, parcels)
        n_sales = _insert_sales(conn, sales_rows, replace=False)
        conn.commit()
        return {"parcels": n_parcels, "sales": n_sales}
    finally:
        conn.close()
