from __future__ import annotations

from meck_agent.config import Settings

# Stored on AIMessage.additional_kwargs so the next Gemini request can
# replay functionCall thought signatures (required for Gemini 3+ tool loops).
FUNCTION_CALL_THOUGHT_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"

# Last-resort validator bypass from Gemini docs when a real signature was dropped.
# https://ai.google.dev/gemini-api/docs/thought-signatures
DUMMY_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"

_THOUGHT_SIGNATURES_PATCHED = False


def is_gemini_3_or_later(model: str | None) -> bool:
    name = (model or "").lower().split("/")[-1]
    if not name.startswith("gemini-"):
        return False
    rest = name[len("gemini-") :]
    major = rest.split(".", 1)[0].split("-", 1)[0]
    return major.isdigit() and int(major) >= 3


def _as_signature_bytes(value: object) -> bytes | None:
    if value is None or value == "":
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        return bytes(value)
    except (TypeError, ValueError):
        return None


def _gemini_model_kwargs(settings: Settings) -> dict:
    """Enable thinking so Gemini 3 emits thought signatures on function calls."""
    kwargs: dict = {
        "model": settings.gemini_model,
        "google_api_key": settings.google_api_key,
        "temperature": 0,
        "include_thoughts": True,
    }
    from langchain_google_genai import ChatGoogleGenerativeAI

    fields = getattr(ChatGoogleGenerativeAI, "model_fields", {}) or {}
    if is_gemini_3_or_later(settings.gemini_model) and "thinking_level" in fields:
        # Gemini 3 Flash still requires thought signatures at "low" / "minimal".
        kwargs["thinking_level"] = "low"
    elif not is_gemini_3_or_later(settings.gemini_model) and "thinking_budget" in fields:
        # 2.5-series knob. Do not send thinking_budget to Gemini 3.
        kwargs["thinking_budget"] = 1024
    return kwargs


def ensure_gemini_thought_signature_support() -> None:
    """Patch older langchain-google-genai so tool loops keep thought signatures.

    langchain-google-genai < 3.1.0 parses Gemini functionCall parts but drops
    `thought_signature`. Gemini 3 then returns HTTP 400 on the next model call
    after a tool runs. Newer library versions already persist and replay them.
    """
    global _THOUGHT_SIGNATURES_PATCHED
    if _THOUGHT_SIGNATURES_PATCHED:
        return

    import inspect

    import langchain_google_genai.chat_models as cm
    from langchain_core.messages import AIMessage

    parse_history_src = inspect.getsource(cm._parse_chat_history)
    if "thought_signature" in parse_history_src:
        _THOUGHT_SIGNATURES_PATCHED = True
        return

    original_parse_response = cm._parse_response_candidate
    original_parse_history = cm._parse_chat_history

    def parse_response_candidate(response_candidate, streaming: bool = False):
        message = original_parse_response(response_candidate, streaming=streaming)
        parts = getattr(getattr(response_candidate, "content", None), "parts", None) or []
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        signatures: dict[str, bytes] = {}
        fc_index = 0
        for part in parts:
            if not getattr(part, "function_call", None):
                continue
            raw_sig = getattr(part, "thought_signature", None)
            sig = _as_signature_bytes(raw_sig)
            if sig and fc_index < len(tool_calls):
                call_id = tool_calls[fc_index].get("id")
                if call_id:
                    signatures[str(call_id)] = sig
            fc_index += 1
        if signatures:
            message.additional_kwargs[FUNCTION_CALL_THOUGHT_SIGNATURES_KEY] = signatures
        return message

    def parse_chat_history(input_messages, convert_system_message_to_human: bool = False):
        system_instruction, history = original_parse_history(
            input_messages, convert_system_message_to_human
        )
        ai_with_tools = [
            msg
            for msg in input_messages
            if isinstance(msg, AIMessage)
            and (getattr(msg, "tool_calls", None) or msg.additional_kwargs.get("function_call"))
        ]
        ai_iter = iter(ai_with_tools)
        for content in history:
            if getattr(content, "role", None) != "model":
                continue
            parts = list(getattr(content, "parts", None) or [])
            if not any(getattr(part, "function_call", None) for part in parts):
                continue
            ai_message = next(ai_iter, None)
            stored = {}
            if ai_message:
                stored = (
                    ai_message.additional_kwargs.get(FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
                    or {}
                )
            first_function_call = True
            tool_index = 0
            for part in parts:
                if not getattr(part, "function_call", None):
                    continue
                if first_function_call and not getattr(part, "thought_signature", None):
                    call_id = None
                    if ai_message and tool_index < len(ai_message.tool_calls):
                        call_id = ai_message.tool_calls[tool_index].get("id")
                    sig = _as_signature_bytes(stored.get(call_id) if call_id else None)
                    part.thought_signature = sig or DUMMY_THOUGHT_SIGNATURE
                    first_function_call = False
                tool_index += 1
        return system_instruction, history

    cm._parse_response_candidate = parse_response_candidate
    cm._parse_chat_history = parse_chat_history
    _THOUGHT_SIGNATURES_PATCHED = True


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

        ensure_gemini_thought_signature_support()
        return ChatGoogleGenerativeAI(**_gemini_model_kwargs(settings))

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
