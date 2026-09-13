from meck_agent.db import epoch_to_iso_date
from meck_agent.ingest_gis import build_cama_where, cama_to_parcel, query_arcgis, sales_to_row
from meck_agent.tools import lookup_parcel_rows, search_sales_rows


def test_epoch_to_iso_date_millis():
    # 2024-03-12 12:00 UTC ~ afternoon ET
    assert epoch_to_iso_date(1_710_244_800_000) == "2024-03-12"


def test_epoch_to_iso_date_string():
    assert epoch_to_iso_date("2024-08-01T00:00:00") == "2024-08-01"
    assert epoch_to_iso_date(None) is None


def test_build_cama_where_zip_and_street():
    where = build_cama_where(zipcode="28202", street="TRYON")
    assert "zipcode LIKE '28202%'" in where
    assert "UPPER(streetname) LIKE '%TRYON%'" in where


def test_cama_and_sales_mapping():
    parcel = cama_to_parcel(
        {
            "parcelid": "12501234",
            "streetname": "oak",
            "loccity": "charlotte",
            "saleprice": 385000,
            "saledate": "2024-03-12",
        }
    )
    assert parcel["parcel_id"] == "12501234"
    assert parcel["street_name"] == "OAK"
    assert parcel["last_sale_date"] == "2024-03-12"

    sale = sales_to_row(
        {"parcelid": "12501234", "saleprice": 100, "saledate": 1_710_244_800_000}
    )
    assert sale["parcel_id"] == "12501234"
    assert sale["sale_date"] == "2024-03-12"


def test_load_sample_csvs(sample_db):
    rows = lookup_parcel_rows(sample_db, "12501234")
    assert len(rows) == 1
    assert rows[0]["address"] == "100 N TRYON ST"
    sales = search_sales_rows(sample_db, street="OAK", year=2024)
    assert len(sales) == 2


def test_query_arcgis_pagination(monkeypatch):
    pages = [
        {
            "features": [{"attributes": {"parcelid": "A"}}, {"attributes": {"parcelid": "B"}}],
            "exceededTransferLimit": True,
        },
        {
            "features": [{"attributes": {"parcelid": "C"}}],
            "exceededTransferLimit": False,
        },
    ]

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class FakeClient:
        def __init__(self):
            self.calls = []

        def post(self, url, data):
            self.calls.append(data)
            return FakeResponse(pages[len(self.calls) - 1])

        def close(self):
            return None

    client = FakeClient()
    rows = query_arcgis(
        "https://example.test/query",
        "1=1",
        "parcelid",
        page_size=2,
        client=client,
    )
    assert [r["parcelid"] for r in rows] == ["A", "B", "C"]
    assert len(client.calls) == 2
