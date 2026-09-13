from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

EXAMPLE_QUESTIONS = [
    "What does NC require when I work with a buyer as a dual agent?",
    "Look up parcel 12501234 and its last recorded sale.",
    "What sold on Oak Street in 2024 according to county records?",
]


def run_app() -> None:
    import streamlit as st

    from meck_agent.config import get_settings
    from meck_agent.status import data_status

    st.set_page_config(
        page_title="Mecklenburg Real Estate Assistant",
        page_icon=":house:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.4rem; max-width: 1100px;}
          .stChatMessage {border-radius: 12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    settings = get_settings()
    status = data_status(settings)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    with st.sidebar:
        st.header("Workspace")
        st.caption("Local SQLite + Chroma. Keys stay in `.env`.")
        st.write(f"**Provider:** `{status['provider']}`")
        st.write(f"**Model:** `{status['model']}`")
        st.write("**API key:** " + ("ready" if status["has_api_key"] else "missing — add it to `.env`"))
        st.write(f"**Parcels:** {status['parcel_count']}")
        st.write(f"**Recorded sales:** {status['sales_count']}")
        st.write("**Guidelines index:** " + ("ready" if status["chroma_ready"] else "not built yet"))

        st.divider()
        st.subheader("Load data")
        if st.button("Load sample parcels & sales", use_container_width=True):
            from meck_agent.ingest_gis import load_sample_csvs

            counts = load_sample_csvs(settings.db_path, settings.sample_dir)
            st.success(f"Loaded {counts['parcels']} parcels and {counts['sales']} sales.")
            st.rerun()

        if st.button("Index sample guidelines", use_container_width=True):
            from meck_agent.rag import ingest_docs

            counts = ingest_docs(
                settings.sample_dir / "docs",
                settings.chroma_path,
                include_remote=False,
                settings=settings,
            )
            st.success(f"Indexed {counts['chunks']} chunks from {counts['source_docs']} sources.")
            st.rerun()

        with st.form("live_gis"):
            st.caption("Live county GIS (attributes only).")
            zipcode = st.text_input("ZIP", value=settings.default_zip)
            street = st.text_input("Street")
            city = st.text_input("City")
            limit = st.number_input("Max parcels", min_value=10, max_value=5000, value=500, step=10)
            live = st.form_submit_button("Download county subset", use_container_width=True)
        if live:
            from meck_agent.ingest_gis import ingest_gis_live

            with st.spinner("Downloading county records..."):
                counts = ingest_gis_live(
                    settings.db_path,
                    zipcode=zipcode or None,
                    street=street or None,
                    city=city or None,
                    limit=int(limit),
                )
            st.success(f"Wrote {counts['parcels']} parcels and {counts['sales']} sales.")
            st.rerun()

        st.divider()
        show_tools = st.checkbox("Show tool calls", value=True)
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

    st.title("Mecklenburg County Real Estate Assistant")
    st.caption(
        "Ask about NC license-law notes or Mecklenburg County recorded tax parcels and sales. "
        "Assessor sales are not MLS comps. This is a study tool, not legal or appraisal advice."
    )

    if not status["has_api_key"]:
        st.warning(
            "No LLM key is loaded. Add `GROQ_API_KEY` or `GOOGLE_API_KEY` to `.env`, "
            "set `LLM_PROVIDER`, then refresh."
        )
    if not status["db_exists"] or status["parcel_count"] == 0:
        st.info("No parcel table yet. Use **Load sample parcels & sales** in the sidebar.")
    if not status["chroma_ready"]:
        st.info("No guideline index yet. Use **Index sample guidelines** in the sidebar.")

    st.write("Try an example:")
    for question in EXAMPLE_QUESTIONS:
        if st.button(question, key=f"ex-{question[:32]}", use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("tool_calls") and show_tools:
                with st.expander("Tool calls"):
                    for call in item["tool_calls"]:
                        st.code(f"{call['name']}({call['args']})")

    typed = st.chat_input("Ask about a PIN, a street sale, or an NC agency rule…")
    question = st.session_state.pending_question or typed
    if question:
        st.session_state.pending_question = None
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if not status["has_api_key"]:
                answer = "Add an API key to `.env` before asking questions."
                tool_calls = []
                st.markdown(answer)
            else:
                from meck_agent.agent import ask_question_detailed

                with st.spinner("Looking up county records and guidelines…"):
                    detail = ask_question_detailed(question, settings)
                answer = detail["answer"] or "The model returned an empty answer."
                tool_calls = detail["tool_calls"]
                st.markdown(answer)
                if tool_calls and show_tools:
                    with st.expander("Tool calls", expanded=True):
                        for call in tool_calls:
                            st.code(f"{call['name']}({call['args']})")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "tool_calls": tool_calls}
        )


if __name__ == "__main__":
    run_app()
