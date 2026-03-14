from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.tools import tool

from commons.agents.llm import llm
from commons.agents.prompt import ARTICLE_SUMMARIZATION_PROMPT_TEMPLATE, QUERY_PLANNER_PROMPT_TEMPLATE
from commons.utils.article import fetch_article_content
from commons.utils.geolocation import get_country_code
from commons.agents.formatter import NewsQuery

from serpapi import GoogleSearch
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ~1,500 tokens of content per article (1 token ≈ 4 chars)
_ARTICLE_CHAR_LIMIT = 6000

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

query_planner_chain = QUERY_PLANNER_PROMPT_TEMPLATE | llm.with_structured_output(NewsQuery)
article_summarization_chain = ARTICLE_SUMMARIZATION_PROMPT_TEMPLATE | llm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_article(item: dict) -> dict:
    """Fetch full content for a single news item."""
    url = item.get("link", "")
    title = item.get("title", "")
    content = fetch_article_content(url)
    print(f"  ✓ Fetched: {title[:70]}")
    return {
        "title": title,
        "url": url,
        "content": content if content else item.get("snippet", "")
    }


def _summarize_single(article: dict, user_query: str, index: int, total: int) -> dict | None:
    """Summarize a single article, returning None if not relevant."""
    content = article.get("content", "")[:_ARTICLE_CHAR_LIMIT]
    result = article_summarization_chain.invoke({
        "user_query": user_query,
        "title": article.get("title", ""),
        "content": content
    })
    summary_text = result.content.strip()
    print(f"  ✓ Summarized ({index}/{total}): {article.get('title', '')[:70]}")
    if summary_text.lower() != "not relevant.":
        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "summary": summary_text
        }
    return None


def _map_summarize_articles(articles: list[dict], user_query: str) -> list[dict]:
    """Map phase: summarize all articles in parallel, drop irrelevant ones."""
    total = len(articles)
    print(f"\n📝 Summarizing {total} articles in parallel...")
    summaries = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_summarize_single, article, user_query, i + 1, total): article
            for i, article in enumerate(articles)
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                summaries.append(result)
    return summaries

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def retrieve_relevant_news(user_input: str) -> list[dict]:
    """
    Fetches and summarizes relevant news articles for a given user question.
    Optimizes the query for Google News, fetches full article content in parallel,
    then runs a parallel map-reduce summarization to return compact per-article summaries.
    """
    # Step 1: Optimize the query for Google News
    print("🧠 Planning search query...")
    news_query: NewsQuery = query_planner_chain.invoke({"user_input": user_input})

    # Step 2: Search Google News via SerpAPI
    print(f"🔎 Searching Google News: \"{news_query.search_query}\"")
    search_results = GoogleSearch({
        "engine": "google_news_light",
        "q": news_query.search_query,
        "gl": get_country_code(),
        "api_key": os.getenv("SERP_API_KEY")
    }).get_dict()
    news_items = search_results.get("news_results", [])

    # Step 3: Fetch full article content in parallel
    print(f"\n🔍 Fetching {len(news_items)} articles in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        articles = list(executor.map(_fetch_article, news_items))

    # Step 4: Map phase — summarize each article in parallel (token-bounded)
    summaries = _map_summarize_articles(articles, news_query.user_query)
    print(f"\n✅ Done — {len(summaries)} relevant articles found.")
    return summaries
