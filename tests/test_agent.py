from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from meck_agent.agent import format_trace, last_ai_text


def test_last_ai_text_skips_thinking_blocks():
    messages = [
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "internal scratch work"},
                {"type": "text", "text": "Dual agency needs written consent."},
            ]
        )
    ]
    assert last_ai_text(messages) == "Dual agency needs written consent."


def test_last_ai_text_skips_empty_tool_calls():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "lookup_parcel", "args": {}, "id": "1"}]),
        ToolMessage(content="{}", tool_call_id="1"),
        AIMessage(content="Parcel 12501234 last sold for $450,000."),
    ]
    assert "450,000" in last_ai_text(messages)


def test_format_trace_includes_tool_name():
    messages = [
        HumanMessage(content="look up 12501234"),
        AIMessage(
            content="",
            tool_calls=[{"name": "lookup_parcel", "args": {"pin_or_address": "12501234"}, "id": "1"}],
        ),
        ToolMessage(content='{"results":[{"parcel_id":"12501234"}]}', tool_call_id="1"),
        AIMessage(content="done"),
    ]
    trace = format_trace(messages)
    assert "[tool call] lookup_parcel" in trace
    assert "[tool result]" in trace
