# ⚡ TL;DR News

A news intelligence chatbot that fetches, extracts, and summarizes live news articles in response to user questions — in any language.

![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C?logo=langchain)
![Groq](https://img.shields.io/badge/Groq-Qwen3--32B-F55036)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)

---

## What it does

Ask anything about current events. TL;DR News searches Google News via SerpAPI, fetches full article content, summarizes each article individually, then synthesizes a structured response with a key insight, summary, and sentiment.

---

## Features

- **Multilingual** — understands and responds in the same language as your query (Indonesian, English, Malay, etc.)
- **Temporal enrichment** — detects recency signals ("hari ini", "latest", "current") and appends date context before searching
- **Localized search** — infers country (`gl`) and language (`hl`) from the query for relevant regional results
- **Map-reduce summarization** — each article is summarized individually, then synthesized into a final answer
- **General news diversification** — broad queries ("what's happening?") fire parallel topic-category searches to avoid single-topic dominance
- **Multi-turn memory** — conversation history is preserved and compacted when it grows long
- **Live progress UI** — Streamlit chat interface shows real-time search and summarization status

---

## Stack

| Layer | Technology |
|---|---|
| LLM | `qwen/qwen3-32b` via Groq (`reasoning_format="hidden"`) |
| Locale classifier | `llama-3.1-8b-instant` via Groq |
| Orchestration | LangChain + LangGraph with `InMemorySaver` checkpointing |
| News search | SerpAPI `google_news_light` engine |
| Article extraction | `trafilatura` (primary), SerpAPI snippet (fallback) |
| UI | Streamlit |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd tldr-news
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
SERP_API_KEY=your_serpapi_key
GROQ_API_KEY=your_groq_key
```

> **Never commit `.env`** — it is listed in `.gitignore`.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Project structure

```
tldr-news/
├── app.py                    # Streamlit chat UI
├── requirements.txt
├── .env.example              # API key template (safe to commit)
├── .env                      # Your actual keys (gitignored)
│
├── src/
│   ├── agent.py              # LangGraph agent with InMemorySaver checkpointing
│   ├── tools.py              # refine_search_query + retrieve_relevant_news tools
│   ├── prompts.py            # All LLM prompts and ChatPromptTemplates
│   ├── models.py             # Pydantic schemas: NewsQuery, LocaleQuery, NewsInsight
│   ├── llm.py                # ChatGroq instances (primary + locale classifier)
│   └── utils/
│       ├── article.py        # Article content fetcher (trafilatura)
│       ├── geolocation.py    # IP-based country/language detection (ipapi.co)
│       ├── logger.py         # File + console logger → logs/
│       └── log_queue.py      # Thread-safe queue for Streamlit live log panel
│
└── logs/                     # Runtime logs (gitignored)
```

---

## Agent pipeline

```
User query
    │
    ▼
[refine_search_query]  ← called when temporal signals detected
    │  Enriches query with current date context
    │
    ▼
[retrieve_relevant_news]
    ├─ query_planner_chain    → optimized search_query + is_general flag
    ├─ locale_detection_chain → gl (country) + hl (language) for SerpAPI
    │
    ├─ SerpAPI search
    │     ├─ Single-topic  (specific queries)
    │     └─ Multi-topic   (general queries — 5 categories in parallel)
    │
    ├─ Article fetch (parallel, up to 8 workers)
    │
    └─ Map-reduce summarization (parallel, 2 workers)
           └─ Per-article summaries → final synthesis by agent
```

---

## Known limitations

- **Paywalled articles** — `trafilatura` may fail; falls back to SerpAPI's snippet field
- **JS-rendered pages** — `requests`-based fetching won't work for SPAs
- **SerpAPI `num` param** — returns ±1 from the requested count
- **Groq rate limits** — summarization retries up to 3× with backoff on `RateLimitError`
- **Memory is in-process** — `InMemorySaver` resets on server restart; no persistent history