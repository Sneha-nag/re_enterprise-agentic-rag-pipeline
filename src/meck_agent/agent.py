from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from meck_agent.config import Settings
from meck_agent.llm import ensure_gemini_thought_signature_support, get_chat_model
from meck_agent.tools import make_tools

SYSTEM_PROMPT = """You are a learning assistant for a North Carolina real estate broker.
You only answer from tools: NC/NCREC guideline excerpts and Mecklenburg County open
tax-parcel / recorded-sale tables.

Rules:
- Call tools before answering factual questions. Do not invent PINs, prices, or rule text.
- Quote parcel ids, sale dates, and prices exactly as returned by lookup_parcel or search_sales.
- Quote guideline text from search_guidelines and name the source title.
- County saleprice/saledate values are recorded/assessor sales, NOT MLS solds. If the user
  asks for comps, say that clearly and mention sales_validity flags.
- If the local subset has no matching row, say so and suggest a broader street/ZIP or a live ingest.
- You are not giving legal, appraisal, or brokerage advice. Suggest verifying current
  NCREC rules and POLARIS/GIS before using anything with a client.
- Stay focused on Mecklenburg County, North Carolina.
"""


def build_agent(
    settings: Settings | None = None,
    *,
    db_path: Path | None = None,
    chroma_path: Path | None = None,
    embeddings=None,
):
    settings = settings or Settings()
    db_path = db_path or settings.db_path
    chroma_path = chroma_path or settings.chroma_path
    if (settings.llm_provider or "").strip().lower() in {"gemini", "google"}:
        # Gemini 3 tool loops must replay the original AIMessage (and its
        # thought signatures). Do not rebuild AIMessages from tool_calls only.
        ensure_gemini_thought_signature_support()
    model = get_chat_model(settings)
    tools = make_tools(db_path, chroma_path, embeddings=embeddings)
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT)


def _message_text(content: object) -> str:
    """Flatten AIMessage content, skipping Gemini thinking blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking" or block.get("thought"):
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return "\n".join(parts)
    return str(content) if content else ""


def last_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = _message_text(message.content)
        if text.strip():
            return text
    return ""


def format_trace(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        name = message.__class__.__name__
        if name == "ToolMessage":
            preview = str(message.content)[:400].replace("\n", " ")
            lines.append(f"[tool result] {preview}")
        elif isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                lines.append(f"[tool call] {call.get('name')} {call.get('args')}")
        elif isinstance(message, HumanMessage):
            lines.append(f"[user] {message.content}")
    return "\n".join(lines)


def collect_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    calls: list[dict] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            calls.append(
                {
                    "name": call.get("name"),
                    "args": call.get("args") or {},
                }
            )
    return calls


def ask_question_detailed(
    question: str,
    settings: Settings | None = None,
    *,
    db_path: Path | None = None,
    chroma_path: Path | None = None,
    embeddings=None,
) -> dict:
    agent = build_agent(
        settings,
        db_path=db_path,
        chroma_path=chroma_path,
        embeddings=embeddings,
    )
    result = agent.invoke({"messages": [("user", question)]})
    messages = result.get("messages") or []
    answer = last_ai_text(messages)
    return {
        "answer": answer,
        "tool_calls": collect_tool_calls(messages),
        "trace": format_trace(messages),
    }


def ask_question(
    question: str,
    settings: Settings | None = None,
    *,
    db_path: Path | None = None,
    chroma_path: Path | None = None,
    embeddings=None,
    verbose: bool = False,
) -> str:
    detail = ask_question_detailed(
        question,
        settings,
        db_path=db_path,
        chroma_path=chroma_path,
        embeddings=embeddings,
    )
    if verbose:
        return f"{detail['trace']}\n\n---\n{detail['answer']}".strip()
    return detail["answer"]
