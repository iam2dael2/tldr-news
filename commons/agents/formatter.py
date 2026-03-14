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
    hl: str = Field(
        description=(
            "BCP-47 language code of the user's query language for Google News host language. "
            "Examples: 'id' for Indonesian, 'en' for English, 'ms' for Malay."
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