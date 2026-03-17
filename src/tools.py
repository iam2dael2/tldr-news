from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.tools import tool

from src.llm import llm, locale_llm
from src.prompts import ARTICLE_SUMMARIZATION_PROMPT_TEMPLATE, LOCALE_DETECTION_PROMPT_TEMPLATE, QUERY_ENRICHMENT_PROMPT_TEMPLATE, QUERY_PLANNER_PROMPT_TEMPLATE
from src.utils.article import fetch_article_content
from src.utils.logger import logger
from src.models import LocaleQuery, NewsQuery

from groq import RateLimitError
from serpapi import GoogleSearch
from datetime import datetime
import os
import time

from src.utils.log_queue import emit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ~750 tokens of content per article (1 token ≈ 4 chars) — enough for lede + key paragraphs
_ARTICLE_CHAR_LIMIT = 3000

# Seconds between successive summarization thread submissions to stay under Groq's TPM limit
_SUMMARIZE_STAGGER_SECS = 10

# Topic modifiers used when diversifying a general news query.
# Each fires a separate search; top 2 results per topic are merged.
_DIVERSITY_TOPICS = [
    "politics diplomacy",
    "economy business finance",
    "technology science",
    "health environment",
    "sports culture",
]

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

locale_detection_chain = LOCALE_DETECTION_PROMPT_TEMPLATE | locale_llm.with_structured_output(LocaleQuery)
query_enrichment_chain = QUERY_ENRICHMENT_PROMPT_TEMPLATE | llm
query_planner_chain = QUERY_PLANNER_PROMPT_TEMPLATE | llm.with_structured_output(NewsQuery)
article_summarization_chain = ARTICLE_SUMMARIZATION_PROMPT_TEMPLATE | llm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serpapi_params(query: str, gl: str, hl: str) -> dict:
    """Build a base SerpAPI params dict."""
    return {
        "engine": "google_news_light",
        "q": query,
        "gl": gl,
        "hl": hl,
        "api_key": os.getenv("SERP_API_KEY"),
    }


def _search_single(query: str, gl: str, hl: str) -> list[dict]:
    """Single-topic search — returns all news_results."""
    return GoogleSearch(_serpapi_params(query, gl, hl)).get_dict().get("news_results", [])


def _search_diverse(base_query: str, gl: str, hl: str) -> list[dict]:
    """
    Multi-topic search for general news queries.
    Fires one search per topic category, takes the top 2 results from each,
    then deduplicates by URL to return a diverse article set.
    """
    seen_urls: set[str] = set()
    diverse_items: list[dict] = []

    def _fetch_topic(topic: str) -> list[dict]:
        return GoogleSearch(
            _serpapi_params(f"{base_query} {topic}", gl, hl)
        ).get_dict().get("news_results", [])

    with ThreadPoolExecutor(max_workers=len(_DIVERSITY_TOPICS)) as executor:
        all_results = executor.map(_fetch_topic, _DIVERSITY_TOPICS)

    for results in all_results:
        for item in results[:2]:
            url = item.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                diverse_items.append(item)

    return diverse_items


def _fetch_article(item: dict) -> dict:
    """Fetch full content for a single news item."""
    url = item.get("link", "")
    title = item.get("title", "")
    content = fetch_article_content(url)
    emit(f"  ✓ {title[:70]}")
    logger.debug(f"Fetched article: {title} | url={url} | content_len={len(content)}")
    return {
        "title": title,
        "url": url,
        "content": content if content else item.get("snippet", "")
    }


def _summarize_single(article: dict, user_query: str, index: int, total: int) -> dict | None:
    """Summarize a single article, returning None if not relevant."""
    title = article.get("title", "")
    content = article.get("content", "")[:_ARTICLE_CHAR_LIMIT]
    for attempt in range(3):
        try:
            result = article_summarization_chain.invoke({
                "user_query": user_query,
                "title": title,
                "content": content
            })
            break
        except RateLimitError:
            wait = 60  # wait for a full TPM window reset (Groq resets on a rolling 60s window)
            emit(f"  ⚠️ Rate limited on summarization, retrying in {wait}s... ({attempt + 1}/3)")
            logger.warning(f"Rate limited on article '{title}' — retrying in {wait}s (attempt {attempt + 1}/3)")
            time.sleep(wait)
    else:
        print(f"  ✗ Skipped (rate limit): {title[:70]}")
        logger.warning(f"Skipped article after 3 rate limit retries: {title}")
        return None

    summary_text = result.content.strip()
    print(f"  ✓ Summarized ({index}/{total}): {title[:70]}")
    logger.debug(f"Summarized ({index}/{total}): {title}")
    if summary_text.lower() != "not relevant.":
        return {"title": title, "url": article.get("url", ""), "summary": summary_text}
    logger.debug(f"Article marked not relevant: {title}")
    return None


def _map_summarize_articles(articles: list[dict], user_query: str) -> list[dict]:
    """Map phase: summarize all articles in parallel, drop irrelevant ones."""
    total = len(articles)
    emit(f"📝 Summarizing {total} articles...")
    logger.info(f"Map phase started: {total} articles to summarize")
    summaries = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for i, article in enumerate(articles):
            if i > 0:
                time.sleep(_SUMMARIZE_STAGGER_SECS)
            futures[executor.submit(_summarize_single, article, user_query, i + 1, total)] = article
        for future in as_completed(futures):
            result = future.result()
            if result:
                summaries.append(result)
    logger.info(f"Map phase complete: {len(summaries)}/{total} articles relevant")
    return summaries

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def refine_search_query(user_input: str) -> str:
    """
    Enriches a user's question with temporal and contextual information to improve search accuracy.
    Call this BEFORE retrieve_relevant_news when the query implies recency or current context
    (e.g., "current tariff", "latest deal", "what is X doing now", or any question about an
    ongoing situation without a specific date). Returns the enriched question as a string.
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    result = query_enrichment_chain.invoke({
        "current_date": current_date,
        "user_input": user_input
    })
    enriched = result.content.strip()
    emit(f"🗓️  Query enriched: \"{enriched}\"")
    logger.info(f"Query enriched: '{user_input}' → '{enriched}'")
    return enriched


@tool
def retrieve_relevant_news(user_input: str) -> list[dict]:
    """
    Fetches and summarizes relevant news articles for a given user question.
    Optimizes the query for Google News, fetches full article content in parallel,
    then runs a parallel map-reduce summarization to return compact per-article summaries.
    For general news queries, fires topic-diversified searches to avoid single-topic dominance.
    """
    logger.info(f"Tool called: retrieve_relevant_news | input='{user_input}'")

    # Step 1: Optimize the query and detect locale in parallel
    emit("🧠 Planning search strategy...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        future_query = ex.submit(query_planner_chain.invoke, {"user_input": user_input})
        future_locale = ex.submit(locale_detection_chain.invoke, {"user_input": user_input})
        news_query: NewsQuery = future_query.result()
        locale: LocaleQuery = future_locale.result()
    gl, hl = locale.gl, locale.hl
    logger.info(f"Query planned: search='{news_query.search_query}' | is_general={news_query.is_general_query} | gl={gl} | hl={hl}")

    # Step 2: Search Google News via SerpAPI
    if news_query.is_general_query:
        emit(f"🔎 General query — searching across {len(_DIVERSITY_TOPICS)} topic categories...")
        logger.info("Search mode: multi-topic diversification")
        news_items = _search_diverse(news_query.search_query, gl, hl)
    else:
        emit(f"🔎 Searching Google News: \"{news_query.search_query}\"")
        logger.info("Search mode: single-topic")
        news_items = _search_single(news_query.search_query, gl, hl)

    logger.info(f"SerpAPI returned {len(news_items)} results total")

    # Step 3: Fetch full article content in parallel
    emit(f"📖 Fetching content from {len(news_items)} articles...")
    logger.info(f"Fetching {len(news_items)} articles in parallel")
    with ThreadPoolExecutor(max_workers=8) as executor:
        articles = list(executor.map(_fetch_article, news_items))

    # Step 4: Map phase — summarize each article in parallel (token-bounded)
    summaries = _map_summarize_articles(articles, news_query.user_query)
    emit(f"📰 Found {len(summaries)} relevant articles")
    logger.info(f"Tool complete: {len(summaries)} relevant summaries returned")

    # Never return an empty list — Groq rejects ToolMessage with content=[]
    if not summaries:
        return "No relevant articles found. The query may be too niche or no recent coverage exists."
    return summaries
