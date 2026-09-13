import json

from meck_agent.tools import lookup_parcel_rows, make_tools, search_sales_rows


def test_lookup_parcel_by_pin_and_address(sample_db):
    by_pin = lookup_parcel_rows(sample_db, "125-012-34")
    assert by_pin[0]["owner_last"] == "SMITH"
    by_addr = lookup_parcel_rows(sample_db, "100 N TRYON")
    assert by_addr[0]["parcel_id"] == "12501234"


def test_search_sales_oak_2024(sample_db):
    rows = search_sales_rows(sample_db, street="Oak", year=2024)
    prices = sorted(r["sale_price"] for r in rows)
    assert prices == [385000, 410000]
    assert all(r["sale_date"].startswith("2024") for r in rows)


def test_search_sales_valid_only_excludes_quitclaim(sample_db):
    all_rows = search_sales_rows(sample_db, street="COLLEGE")
    valid = search_sales_rows(sample_db, street="COLLEGE", valid_only=True)
    assert len(all_rows) == 2
    assert len(valid) == 1
    assert valid[0]["sales_validity"] == "Y"


def test_tools_json_roundtrip(sample_db, sample_chroma, hash_embeddings):
    tools = {t.name: t for t in make_tools(sample_db, sample_chroma, embeddings=hash_embeddings)}
    parcel = json.loads(tools["lookup_parcel"].invoke({"pin_or_address": "12501234"}))
    assert parcel["results"][0]["city"] == "CHARLOTTE"
    sales = json.loads(
        tools["search_sales"].invoke({"street": "OAK", "year": 2024, "zipcode": "28205"})
    )
    assert len(sales["results"]) == 2
    assert "not MLS" in sales["disclaimer"]
    guides = json.loads(
        tools["search_guidelines"].invoke({"query": "dual agency written consent"})
    )
    assert guides["results"]
    blob = " ".join(hit["excerpt"].lower() for hit in guides["results"])
    assert "dual" in blob
