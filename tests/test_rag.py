from meck_agent.rag import retrieve


def test_guidelines_retrieve_dual_agency(sample_chroma, hash_embeddings):
    docs = retrieve(
        sample_chroma,
        "What does North Carolina require for dual agency consent?",
        k=4,
        embeddings=hash_embeddings,
    )
    text = "\n".join(doc.page_content.lower() for doc in docs)
    assert "dual agency" in text
    assert "consent" in text


def test_guidelines_retrieve_license_law(sample_chroma, hash_embeddings):
    docs = retrieve(
        sample_chroma,
        "Is a real estate broker license required in North Carolina?",
        k=4,
        embeddings=hash_embeddings,
    )
    text = "\n".join(doc.page_content.lower() for doc in docs)
    assert "license" in text
    assert "93a" in text or "broker" in text
