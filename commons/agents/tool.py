from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.tools import tool

from commons.agents.llm import llm
from commons.agents.prompt import ARTICLE_SUMMARIZATION_PROMPT_TEMPLATE, QUERY_ENRICHMENT_PROMPT_TEMPLATE, QUERY_PLANNER_PROMPT_TEMPLATE
from commons.utils.article import fetch_article_content
from commons.utils.geolocation import get_country_code
from commons.utils.logger import logger
from commons.agents.formatter import NewsQuery

from groq import RateLimitError
from serpapi import GoogleSearch
from datetime import datetime
import os
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ~1,500 tokens of content per article (1 token ≈ 4 chars)
_ARTICLE_CHAR_LIMIT = 6000

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

    for topic in _DIVERSITY_TOPICS:
        topic_query = f"{base_query} {topic}"
        results = GoogleSearch(_serpapi_params(topic_query, gl, hl)).get_dict().get("news_results", [])
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
    print(f"  ✓ Fetched: {title[:70]}")
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
            wait = 10 * (attempt + 1)  # 10s, 20s, 30s
            print(f"  ⚠️ Rate limited, retrying in {wait}s... ({attempt + 1}/3)")
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
    print(f"\n📝 Summarizing {total} articles in parallel...")
    logger.info(f"Map phase started: {total} articles to summarize")
    summaries = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_summarize_single, article, user_query, i + 1, total): article
            for i, article in enumerate(articles)
        }
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
    print(f"🗓️  Query enriched: \"{enriched}\"")
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

    # Step 1: Optimize the query for Google News
    print("🧠 Planning search query...")
    news_query: NewsQuery = query_planner_chain.invoke({"user_input": user_input})
    gl = get_country_code()
    hl = news_query.hl
    logger.info(f"Query planned: search='{news_query.search_query}' | is_general={news_query.is_general_query} | gl={gl} | hl={hl}")

    # Step 2: Search Google News via SerpAPI
    if news_query.is_general_query:
        print(f"🔎 General query detected — searching across {len(_DIVERSITY_TOPICS)} topic categories...")
        logger.info("Search mode: multi-topic diversification")
        news_items = _search_diverse(news_query.search_query, gl, hl)
    else:
        print(f"🔎 Searching Google News: \"{news_query.search_query}\"")
        logger.info("Search mode: single-topic")
        news_items = _search_single(news_query.search_query, gl, hl)

    logger.info(f"SerpAPI returned {len(news_items)} results total")

    # Step 3: Fetch full article content in parallel
    print(f"\n🔍 Fetching {len(news_items)} articles in parallel...")
    logger.info(f"Fetching {len(news_items)} articles in parallel")
    with ThreadPoolExecutor(max_workers=8) as executor:
        articles = list(executor.map(_fetch_article, news_items))

    # Step 4: Map phase — summarize each article in parallel (token-bounded)
    summaries = _map_summarize_articles(articles, news_query.user_query)
    print(f"\n✅ Done — {len(summaries)} relevant articles found.")
    logger.info(f"Tool complete: {len(summaries)} relevant summaries returned")
    return summaries
