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

_ARTICLE_CHAR_LIMIT = 6000 # ~1,500 tokens of content per article (1 token ≈ 4 chars)

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

query_planner_chain = QUERY_PLANNER_PROMPT_TEMPLATE | llm.with_structured_output(NewsQuery)
article_summarization_chain = ARTICLE_SUMMARIZATION_PROMPT_TEMPLATE | llm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_summarize_articles(articles: list[dict], user_query: str) -> list[dict]:
    """Map phase: summarize each article individually to keep each LLM call token-bounded."""
    summaries = []
    for article in articles:
        content = article.get("content", "")[:_ARTICLE_CHAR_LIMIT]
        result = article_summarization_chain.invoke({
            "user_query": user_query,
            "title": article.get("title", ""),
            "content": content
        })
        summary_text = result.content.strip()
        if summary_text.lower() != "not relevant.":
            summaries.append({
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "summary": summary_text
            })
    return summaries

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def retrieve_relevant_news(user_input: str) -> list[dict]:
    """
    Fetches and summarizes relevant news articles for a given user question.
    Optimizes the query for Google News, fetches full article content,
    then runs a map-reduce summarization to return compact per-article summaries.
    """
    # Step 1: Optimize the query for Google News
    news_query: NewsQuery = query_planner_chain.invoke({"user_input": user_input})

    # Step 2: Search Google News via SerpAPI
    search_results = GoogleSearch({
        "engine": "google_news_light",
        "q": news_query.search_query,
        "gl": get_country_code(),
        "api_key": os.getenv("SERP_API_KEY")
    }).get_dict()
    news_items = search_results.get("news_results", [])

    # Step 3: Fetch full article content (fallback to snippet if extraction fails)
    articles = []
    for item in news_items:
        url = item.get("link", "")
        content = fetch_article_content(url)
        articles.append({
            "title": item.get("title", ""),
            "url": url,
            "content": content if content else item.get("snippet", "")
        })

    # Step 4: Map phase — summarize each article individually (token-bounded)
    return _map_summarize_articles(articles, news_query.user_query)
