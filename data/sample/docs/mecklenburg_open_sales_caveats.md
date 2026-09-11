# Mecklenburg tax-sale records vs MLS (study notes)

Source: Mecklenburg County GIS / Land Records open data (Tax Parcels with CAMA, Tax Parcel Sales).
URLs:
- https://meckgis.mecklenburgcountync.gov/server/rest/services/TaxParcel_camadata/MapServer
- https://meckgis.mecklenburgcountync.gov/server/rest/services/TaxParcelSales/FeatureServer
- https://maps.mecklenburgcountync.gov/opendata/metadata/Tax_Parcels_with_CAMA_Data.html

## What the open data contains

County CAMA and Tax Parcel Sales layers include parcel identifiers, situs-style addresses, land use, assessed/market values, and recorded sale dates and prices processed from Register of Deeds instruments. A `salesvalidity` / valid-sale flag indicates whether the assessor treated the transfer as a usable market sale.

## What it is not

These are **not MLS closed sales**. They can include foreclosure-related deeds, related-party transfers, quitclaims, and multi-parcel or vacant-land deals. Days on market, list price, concessions, interior condition, and agent remarks are not in the open tax layers.

When a user asks for "comps," answer from county recorded sales and say clearly that MLS comparables may differ. Prefer sales flagged valid when discussing market evidence.

## Subset ingest

This learning project defaults to a ZIP or street subset (for example 28202) so a laptop can download quickly. Mailing zip on CAMA may differ from the property location; street and city filters are often more reliable.
