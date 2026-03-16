import html as _html
import markdown as _markdown
import queue as _queue
import re
import threading
import uuid

import streamlit as st
from langchain_core.messages import RemoveMessage, SystemMessage

from src.agent import agent
from src.llm import llm
from src.utils import log_queue as lq
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TL;DR News",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS — dark mode, Perplexity-inspired
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Hide Streamlit chrome ─────────────────────────────────────────────── */
#MainMenu, header, footer { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

/* ── Global ────────────────────────────────────────────────────────────── */
.stApp { background-color: #0D0D0D; color: #E5E5E5; }
.block-container { max-width: 760px; padding-top: 1.5rem; padding-bottom: 6rem; }

/* ── Header ────────────────────────────────────────────────────────────── */
.tldr-header     { text-align: center; padding: 1.5rem 0 1rem; }
.tldr-title      { font-size: 1.9rem; font-weight: 700; color: #E5E5E5; letter-spacing: -0.5px; }
.tldr-title span { color: #4ADE80; }
.tldr-tagline    { font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem; }

/* ── Chat message wrappers ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── User bubble ───────────────────────────────────────────────────────── */
.user-bubble {
    background-color: #1A2E1A;
    border: 1px solid #2D4A2D;
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1.1rem;
    font-size: 0.9rem;
    color: #E5E5E5;
    line-height: 1.6;
    display: inline-block;
    max-width: 80%;
    float: right;
    clear: both;
    margin-bottom: 0.25rem;
}

/* ── News card (structured response) ──────────────────────────────────── */
.news-card {
    background-color: #1A1A1A;
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 0.25rem;
    clear: both;
}
.news-card-insight {
    font-size: 0.975rem;
    font-weight: 600;
    color: #E5E5E5;
    line-height: 1.55;
    margin-bottom: 0.85rem;
}
.news-card-summary {
    font-size: 0.855rem;
    color: #A3A3A3;
    line-height: 1.75;
    margin-bottom: 0.85rem;
}
.news-card-footer {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    border-top: 1px solid #262626;
    padding-top: 0.75rem;
    margin-top: 0.25rem;
}
.sentiment-reason {
    font-size: 0.75rem;
    color: #6B7280;
}

/* ── Plain response card ───────────────────────────────────────────────── */
.plain-card {
    background-color: #1A1A1A;
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 0.875rem;
    color: #D4D4D4;
    line-height: 1.75;
    clear: both;
}

/* ── Sentiment badges ──────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}
.badge-positive { background-color: #052e16; color: #4ADE80; border: 1px solid #166534; }
.badge-negative { background-color: #450a0a; color: #F87171; border: 1px solid #7f1d1d; }
.badge-neutral  { background-color: #1c1c1c; color: #9CA3AF; border: 1px solid #374151; }

/* ── Chat input ────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background-color: #1A1A1A !important;
    color: #E5E5E5 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #4ADE80 !important;
    box-shadow: 0 0 0 1px #4ADE80 !important;
}
[data-testid="stChatInputContainer"] {
    background-color: #0D0D0D !important;
    padding-bottom: 1rem !important;
}

/* ── Status widget ─────────────────────────────────────────────────────── */
[data-testid="stStatusWidget"] {
    background-color: #1A1A1A !important;
    border: 1px solid #262626 !important;
    border-radius: 8px !important;
}

/* ── Divider between turns ─────────────────────────────────────────────── */
.turn-spacer { height: 1.25rem; clear: both; }

/* ── Live log panel ────────────────────────────────────────────────────── */
.log-panel {
    font-size: 0.8rem;
    line-height: 1.9;
    padding: 0.1rem 0;
}
.log-main {
    color: #C4C4C4;
}
.log-sub {
    color: #6B7280;
    padding-left: 1.1rem;
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COMPACT_THRESHOLD = 10
_WELCOME_MSG = (
    "👋 Welcome to **⚡ TL;DR News**! I fetch and summarize the latest news for you.\n\n"
    "Try asking: *\"Apa berita terkini hari ini?\"* or *\"What's the latest on US tariffs?\"*"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    """Try to extract structured fields from a formatted news response."""
    ki    = re.search(r"🔑\s*KEY INSIGHT:\s*(.+?)(?=\n\n|📰|$)", text, re.DOTALL)
    sm    = re.search(r"📰\s*SUMMARY:\s*\n(.+?)(?=\n\n📊|\n📊|$)", text, re.DOTALL)
    sent  = re.search(r"📊\s*SENTIMENT:\s*\*{0,2}(POSITIVE|NEGATIVE|NEUTRAL)\*{0,2}", text, re.IGNORECASE)
    rsn   = re.search(r"Reason:\s*(.+?)(?=\n|$)", text)

    if ki and sm and sent:
        return {
            "structured": True,
            "key_insight": ki.group(1).strip(),
            "summary":     sm.group(1).strip(),
            "sentiment":   sent.group(1).upper(),
            "reason":      rsn.group(1).strip() if rsn else "",
            "raw":         text,
        }
    return {"structured": False, "raw": text}


def _render_message(parsed: dict) -> None:
    """Render an assistant message — structured card or plain text."""
    if parsed["structured"]:
        sentiment   = parsed["sentiment"]
        badge_class = {"POSITIVE": "badge-positive", "NEGATIVE": "badge-negative"}.get(sentiment, "badge-neutral")
        reason_html = f'<span class="sentiment-reason">— {parsed["reason"]}</span>' if parsed["reason"] else ""
        st.markdown(f"""
<div class="news-card">
  <div class="news-card-insight">🔑 {_markdown.markdown(parsed["key_insight"])}</div>
  <div class="news-card-summary">{_markdown.markdown(parsed["summary"])}</div>
  <div class="news-card-footer">
    <span class="badge {badge_class}">{sentiment}</span>
    {reason_html}
  </div>
</div>""", unsafe_allow_html=True)
    else:
        # Render markdown inside a styled card
        st.markdown(
            f'<div class="plain-card">{_markdown.markdown(parsed["raw"])}</div>',
            unsafe_allow_html=True
        )


def _compact_if_needed(config: dict) -> None:
    """Summarize old messages when history grows too long."""
    state    = agent.get_state(config)
    messages = state.values.get("messages", [])
    if len(messages) <= _COMPACT_THRESHOLD:
        return
    to_summarize  = messages[:-2]
    history_text  = "\n".join(f"{m.type}: {m.content[:400]}" for m in to_summarize)
    summary       = llm.invoke(
        f"Summarize this conversation concisely, preserving all key facts:\n\n{history_text}"
    ).content
    remove_ops    = [RemoveMessage(id=m.id) for m in to_summarize]
    summary_msg   = SystemMessage(content=f"[Conversation Summary]\n{summary}")
    agent.update_state(config, {"messages": remove_ops + [summary_msg]})
    logger.info(f"Compaction: {len(to_summarize)} messages condensed into 1 summary")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:          # display history
    st.session_state.messages = []

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tldr-header">
  <div class="tldr-title">⚡ TL;DR <span>News</span></div>
  <div class="tldr-tagline">Ask anything. Get the news that matters.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

# Welcome message on first load
if not st.session_state.messages:
    with st.chat_message("assistant"):
        _render_message({"structured": False, "raw": _WELCOME_MSG})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            _render_message(msg["parsed"])
    st.markdown('<div class="turn-spacer"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input + agent invocation
# ---------------------------------------------------------------------------
if query := st.chat_input("Ask about any news topic…"):

    # ── Show user bubble ────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-bubble">{query}</div>', unsafe_allow_html=True)
    st.markdown('<div class="turn-spacer"></div>', unsafe_allow_html=True)
    logger.info(f"User query: '{query}'")

    # ── Invoke agent with live progress ────────────────────────────────
    with st.chat_message("assistant"):
        n_before = len(agent.get_state(config).values.get("messages", []))
        _result  = {"text": "", "error": None}

        # Wire up the log queue for this request
        msg_queue: _queue.Queue = _queue.Queue()
        lq.set_active_queue(msg_queue)

        def _run_agent() -> None:
            try:
                for update in agent.stream(
                    {"messages": [("user", query)]},
                    config=config,
                    stream_mode="updates",
                ):
                    msg_queue.put(("_UPDATE", update))
            except Exception as exc:
                _result["error"] = exc
            finally:
                msg_queue.put(("_DONE", None))

        agent_thread = threading.Thread(target=_run_agent, daemon=True)

        with st.status("🔍 Thinking…", expanded=True) as status:
            log_placeholder = st.empty()
            log_lines: list[str] = []

            def _render_log() -> None:
                if not log_lines:
                    return
                items_html = ""
                for raw in log_lines:
                    # strip leading newlines (used for terminal spacing)
                    line = raw.lstrip("\n")
                    is_sub = line.startswith("  ")
                    text   = _html.escape(line.strip())
                    css    = "log-sub" if is_sub else "log-main"
                    items_html += f'<div class="{css}">{text}</div>'
                log_placeholder.markdown(
                    f'<div class="log-panel">{items_html}</div>',
                    unsafe_allow_html=True,
                )

            def _add_log(line: str) -> None:
                log_lines.append(line)
                _render_log()

            agent_thread.start()

            while True:
                try:
                    item = msg_queue.get(timeout=0.05)
                except _queue.Empty:
                    continue

                kind, data = item[0], item[1]

                if kind == "_DONE":
                    break

                elif kind == "_LOG":
                    _add_log(data)

                elif kind == "_UPDATE":
                    for node_name, node_data in data.items():
                        msgs = (node_data.get("messages", [])
                                if isinstance(node_data, dict) else [])

                        if node_name == "model":
                            for m in msgs:
                                for tc in getattr(m, "tool_calls", []):
                                    if tc["name"] == "refine_search_query":
                                        _add_log("🗓️ Enriching query with temporal context…")
                                    elif tc["name"] == "retrieve_relevant_news":
                                        arg = tc.get("args", {}).get("user_input", "")
                                        _add_log(f"📡 Retrieving news for: \"{arg[:80]}\"")
                                if (getattr(m, "content", "")
                                        and not getattr(m, "tool_calls", [])):
                                    _result["text"] = m.content
                                    _add_log("✍️ Generating answer…")

            agent_thread.join()
            lq.clear_active_queue()

            # Fallback: read from graph state if stream missed the response
            if not _result["text"]:
                final_msgs = agent.get_state(config).values.get("messages", [])
                if final_msgs:
                    _result["text"] = final_msgs[-1].content

            if _result["error"]:
                raise _result["error"]

            status.update(label="✅ Done", state="complete", expanded=False)

        response_text = _result["text"]

        # ── Log decision ────────────────────────────────────────────────
        new_msgs    = agent.get_state(config).values.get("messages", [])[n_before:]
        tool_called = any(getattr(m, "type", "") == "tool" for m in new_msgs)
        logger.info(
            "Decision: External retrieval triggered"
            if tool_called else
            "Decision: Answered from conversation context / general knowledge"
        )

        # ── Render response ─────────────────────────────────────────────
        parsed = _parse_response(response_text)
        _render_message(parsed)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": response_text,
            "parsed":  parsed,
        })

    st.markdown('<div class="turn-spacer"></div>', unsafe_allow_html=True)

    # ── Compact if needed ───────────────────────────────────────────────
    _compact_if_needed(config)
