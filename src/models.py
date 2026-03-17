from pydantic import BaseModel, Field, confloat
from enum import Enum


class NewsQuery(BaseModel):
    """Structured Input for Generating News Insights."""
    search_query: str = Field(
        description="Optimized search query for Google News (keywords only, no filler words)"
    )
    user_query: str = Field(
        description="Original user question, used for summarization context"
    )
    is_general_query: bool = Field(
        description=(
            "True if the user is asking for general/broad news with no specific topic "
            "(e.g. 'latest news today', 'berita terkini', 'what's happening'). "
            "False if the user is asking about a specific topic, event, or entity."
        )
    )
    suggest_videos: bool = Field(
        description=(
            "True if the topic would genuinely benefit from YouTube video recommendations — "
            "e.g. major events or crises (war, disaster, election), complex topics that benefit "
            "from visual explanation (economic policy, geopolitical conflict, scientific discovery), "
            "or events where footage adds real context (protests, speeches, natural disasters). "
            "False for simple factual questions, very recent breaking news where no good video "
            "exists yet, or topics already well-covered by text alone."
        )
    )


class LocaleQuery(BaseModel):
    """Locale parameters inferred from the user's query for SerpAPI."""
    gl: str = Field(
        description=(
            "SerpAPI Google country code (gl) inferred from the query's language, topic, or geographic context. "
            "Must be a valid ISO 3166-1 alpha-2 code supported by SerpAPI (e.g. 'id', 'us', 'gb', 'sg', 'my'). "
            "Default to 'us' if country context cannot be determined."
        )
    )
    hl: str = Field(
        description=(
            "SerpAPI Google language code (hl) matching the query's language. "
            "Must be a valid BCP-47 code supported by SerpAPI (e.g. 'id', 'en', 'ms', 'zh-cn'). "
            "Default to 'en' if language cannot be determined."
        )
    )


class Sentiment(str, Enum):
    """Overall sentiment of the aggregated news."""
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class NewsInsight(BaseModel):
    """Structured insight generated from aggregated news articles about a topic."""

    answer: str = Field(
        description="A concise summarized answer of the key developments, based on the aggregated news articles."
    )

    sentiment: Sentiment = Field(
        description="The overall sentiment of the news coverage regarding the topic."
    )

    importance_score: confloat(ge=0, le=1) = Field(
        description="A score from 0 to 1 indicating how important or impactful the news topic is relative to other current events."
    )