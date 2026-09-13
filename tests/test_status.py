from meck_agent.config import Settings
from meck_agent.ingest_gis import load_sample_csvs
from meck_agent.status import data_status, provider_model_name


def test_data_status_counts_sample_rows(tmp_path, sample_dir):
    settings = Settings(meck_home=tmp_path, llm_provider="gemini", google_api_key="x")
    load_sample_csvs(settings.db_path, sample_dir)
    status = data_status(settings)
    assert status["parcel_count"] == 6
    assert status["sales_count"] == 7
    assert status["db_exists"] is True
    assert status["has_api_key"] is True
    assert status["chroma_ready"] is False


def test_provider_model_name_gemini():
    settings = Settings(llm_provider="gemini", gemini_model="gemini-3.6-flash")
    assert provider_model_name(settings) == "gemini-3.6-flash"
