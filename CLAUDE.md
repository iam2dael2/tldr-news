# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

TL;DR News is a news intelligence chatbot. Users ask questions about current events and receive structured insights (summary, sentiment, importance score) synthesized from live news articles retrieved via SerpAPI.

## Stack

- **LLM**: Qwen `qwen/qwen3-32b` via Groq (`langchain-groq`), `reasoning_format="hidden"`, `temperature=0`, `timeout=10`, `max_retries=2`
- **Orchestration**: LangChain + LangGraph (`langgraph`) with `InMemorySaver` checkpointing
- **News search**: SerpAPI `google_news_light` engine (`serpapi`)
- **Article extraction**: `newspaper3k` (primary), SerpAPI snippet field (fallback for paywalled/JS-rendered pages)
- **UI**: Streamlit (planned — not yet implemented)
- **Config**: `python-dotenv` — keys loaded from `.env`

## Environment Variables

```
SERP_API_KEY=...
GROQ_API_KEY=...
```

See `.env.example` for the template.

## Running

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Run the SerpAPI search demo
python main.py

# Run the LangChain runnable playground
python test.py
```

No test suite or build step exists yet.

## Architecture

The system uses a **two-stage pipeline**:

1. **Query Planner** (`commons/agents/tool.py`): Takes raw user input and uses a structured-output LLM chain to produce a `NewsQuery` — an optimized `search_query` for SerpAPI and the original `user_query` for summarization context.

2. **Agent** (`commons/agents/model.py`): A LangGraph agent with `retrieve_relevant_news` as its only tool. The agent decides when to call the tool based on whether the question requires live news (see `AGENT_SYSTEM_PROMPT` in `prompt.py`). Returns a structured `NewsInsight` via `ToolStrategy`.

### Summarization Strategy

Uses a **map-reduce** approach (not "stuff all at once"):
1. Summarize each article individually (truncated to ~1,500–2,000 tokens each)
2. Synthesize individual summaries into a final answer

This keeps each LLM call context-bounded and scales to many articles.

### Key Data Models (`commons/agents/formatter.py`)

- `NewsQuery`: `search_query` (keyword-only, filler removed) + `user_query` (original, preserved verbatim)
- `NewsInsight`: `answer` (str, ≤150 words) + `sentiment` (Sentiment enum) + `importance_score` (float 0–1)
- `Sentiment`: enum of `positive`, `neutral`, `negative`

### Geolocation (`commons/utils/geolocation.py`)

Calls `ipapi.co/country/` to determine the user's country code, passed as the `gl` parameter to SerpAPI for localized results. Falls back to `"us"` on failure.

### Prompts (`commons/agents/prompt.py`)

- `AGENT_SYSTEM_PROMPT`: Controls decision logic — answer from knowledge, call tool for recent news, or decline out-of-scope requests
- `NEWS_QUERY_PLANNER_SYSTEM_PROMPT`: Strips filler words, preserves named entities and time references; responds in the same language as the user

### Agent Output Format (when summarizing news)

```
🔑 KEY INSIGHT: <one sentence>

📰 SUMMARY:
<max 150 words, plain English, no jargon>

📊 SENTIMENT: <POSITIVE | NEGATIVE | NEUTRAL>
Reason: <one sentence>
```

### Agent Decision Logic

1. General knowledge / not news-related → answer directly, no tool call
2. Recent events / breaking news → call `retrieve_relevant_news`, then summarize
3. Out of scope (image gen, games, etc.) → politely decline

### Agent Rules

- Be objective — no editorializing
- Always respond in English regardless of input language
- Ground summaries strictly in retrieved articles — never fabricate
- If sources conflict, acknowledge the discrepancy
- If no results returned, tell the user and suggest rephrasing

## External Services

| Service | Purpose | Env var |
|---|---|---|
| Groq API | LLM inference (`qwen/qwen3-32b`) | `GROQ_API_KEY` |
| SerpAPI | Google News search | `SERP_API_KEY` |
| ipapi.co | IP-based country detection | — |

SerpAPI params of note: `engine=google_news_light`, `gl` (country), `hl` (language), `num` (result hint, not guaranteed exact).

## Known Limitations

- **Paywalled articles**: `newspaper3k` may fail; fall back to SerpAPI's `snippet` field
- **JS-rendered pages**: `requests` + BeautifulSoup won't work for SPAs; `playwright` or `selenium` needed for those
- **Rate limiting**: Fetching multiple URLs in parallel can trigger throttling — use asyncio with concurrency limits or brief delays
- **SerpAPI `num` param**: Returns ±1 from the requested count; don't hard-depend on an exact number

## Planned Work

- [ ] Streamlit chat UI
- [ ] Full article fetch + map-reduce summarization pipeline
- [ ] Multi-turn conversation memory
- [ ] `test.py` moved to `scratch/` or removed (it's a throwaway LangChain experiment)
