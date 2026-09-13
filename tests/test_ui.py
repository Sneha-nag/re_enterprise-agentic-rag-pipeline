from meck_agent.ui import EXAMPLE_QUESTIONS


def test_example_questions_cover_rag_and_sql():
    blob = " ".join(EXAMPLE_QUESTIONS).lower()
    assert "dual agent" in blob
    assert "12501234" in blob
    assert "oak street" in blob
