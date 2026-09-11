from langchain_core.messages import AIMessage, HumanMessage

from meck_agent.config import Settings
from meck_agent.llm import (
    DUMMY_THOUGHT_SIGNATURE,
    FUNCTION_CALL_THOUGHT_SIGNATURES_KEY,
    _gemini_model_kwargs,
    ensure_gemini_thought_signature_support,
    is_gemini_3_or_later,
)


def test_is_gemini_3_or_later():
    assert is_gemini_3_or_later("gemini-3.6-flash")
    assert is_gemini_3_or_later("models/gemini-3-pro-preview")
    assert not is_gemini_3_or_later("gemini-2.0-flash")
    assert not is_gemini_3_or_later("gemini-2.5-flash")


def test_gemini_init_enables_thoughts_without_budget_on_v3():
    settings = Settings(
        gemini_model="gemini-3.6-flash",
        google_api_key="test-key",
    )
    kwargs = _gemini_model_kwargs(settings)
    assert kwargs["include_thoughts"] is True
    assert "thinking_budget" not in kwargs


def test_parse_history_attaches_thought_signature():
    ensure_gemini_thought_signature_support()
    from langchain_google_genai.chat_models import _parse_chat_history

    real_sig = b"real-thought-signature"
    messages = [
        HumanMessage(content="Look up parcel 12501234"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "lookup_parcel",
                    "args": {"pin_or_address": "12501234"},
                    "id": "call-1",
                }
            ],
            additional_kwargs={FUNCTION_CALL_THOUGHT_SIGNATURES_KEY: {"call-1": real_sig}},
        ),
    ]
    _system, history = _parse_chat_history(messages)
    function_parts = [
        part
        for content in history
        for part in content.parts
        if getattr(part, "function_call", None)
    ]
    assert function_parts
    assert function_parts[0].thought_signature == real_sig


def test_parse_history_uses_dummy_signature_when_missing():
    ensure_gemini_thought_signature_support()
    from langchain_google_genai.chat_models import _parse_chat_history

    messages = [
        HumanMessage(content="What sold on Oak Street?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_sales", "args": {"street": "OAK"}, "id": "call-2"}
            ],
        ),
    ]
    _system, history = _parse_chat_history(messages)
    function_parts = [
        part
        for content in history
        for part in content.parts
        if getattr(part, "function_call", None)
    ]
    assert function_parts
    assert function_parts[0].thought_signature == DUMMY_THOUGHT_SIGNATURE


def test_parse_response_stores_function_call_signature():
    ensure_gemini_thought_signature_support()
    from google.ai.generativelanguage_v1beta.types import Candidate, Content, FunctionCall, Part
    from langchain_google_genai.chat_models import _parse_response_candidate

    part = Part(
        function_call=FunctionCall(name="lookup_parcel", args={"pin_or_address": "1"}),
        thought_signature=b"from-gemini",
    )
    candidate = Candidate(content=Content(role="model", parts=[part]))
    message = _parse_response_candidate(candidate, streaming=False)
    stored = message.additional_kwargs.get(FUNCTION_CALL_THOUGHT_SIGNATURES_KEY) or {}
    assert list(stored.values()) == [b"from-gemini"]
