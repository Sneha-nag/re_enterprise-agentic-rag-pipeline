from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

PARCEL_FIELDS = (
    "parcel_id",
    "nc_pin",
    "address",
    "street_number",
    "street_name",
    "city",
    "zipcode",
    "neighborhood",
    "land_use",
    "land_use_description",
    "acres",
    "heated_area",
    "bedrooms",
    "year_built",
    "total_value",
    "market_value",
    "last_sale_price",
    "last_sale_date",
    "valid_sale",
    "owner_last",
    "owner_first",
)

SALE_FIELDS = (
    "parcel_id",
    "sale_price",
    "sale_date",
    "sales_validity",
    "sold_as_vacant",
    "land_use",
    "land_use_description",
    "grantor",
    "grantee",
    "deed_description",
    "legal_reference",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS parcels (
    parcel_id TEXT PRIMARY KEY,
    nc_pin TEXT,
    address TEXT,
    street_number TEXT,
    street_name TEXT,
    city TEXT,
    zipcode TEXT,
    neighborhood TEXT,
    land_use TEXT,
    land_use_description TEXT,
    acres REAL,
    heated_area REAL,
    bedrooms INTEGER,
    year_built INTEGER,
    total_value REAL,
    market_value REAL,
    last_sale_price REAL,
    last_sale_date TEXT,
    valid_sale TEXT,
    owner_last TEXT,
    owner_first TEXT
);

CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parcel_id TEXT,
    sale_price REAL,
    sale_date TEXT,
    sales_validity TEXT,
    sold_as_vacant TEXT,
    land_use TEXT,
    land_use_description TEXT,
    grantor TEXT,
    grantee TEXT,
    deed_description TEXT,
    legal_reference TEXT
);

CREATE INDEX IF NOT EXISTS idx_sales_parcel ON sales(parcel_id);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_parcels_street ON parcels(street_name);
CREATE INDEX IF NOT EXISTS idx_parcels_zip ON parcels(zipcode);
CREATE INDEX IF NOT EXISTS idx_parcels_address ON parcels(address);
"""


def connect(db_path: Path):
    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def epoch_to_iso_date(value: object) -> str | None:
    """Convert ArcGIS epoch (ms or s) or a date-like string to YYYY-MM-DD."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text[:10].count("-") == 2 and len(text) >= 10:
            return text[:10]
        try:
            as_num = float(text)
        except ValueError:
            return text[:10]
        value = as_num
    if isinstance(value, (int, float)):
        ms = float(value)
        if ms > 1e11:
            seconds = ms / 1000.0
        else:
            seconds = ms
        return datetime.fromtimestamp(seconds, tz=EASTERN).date().isoformat()
    return str(value)


def compact_id(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum()).upper()
