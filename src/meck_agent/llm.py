from __future__ import annotations

from meck_agent.config import Settings


def get_chat_model(settings: Settings | None = None):
    """Return a LangChain chat model for Groq, Gemini, or OpenAI."""
    settings = settings or Settings()
    provider = (settings.llm_provider or "groq").strip().lower()

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. "
                "Copy .env.example to .env and add a free key from https://console.groq.com/"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )

    if provider in {"gemini", "google"}:
        if not settings.google_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is empty. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use groq, gemini, or openai."
    )
