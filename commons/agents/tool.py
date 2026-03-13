from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from commons.agents.prompt import NEWS_QUERY_PLANNER_SYSTEM_PROMPT
from commons.agents.formatter import NewsQuery

from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM instance using Groq with the Qwen model
llm: BaseChatModel = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="hidden",
    timeout=10,
    max_retries=2
)

query_planner_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", NEWS_QUERY_PLANNER_SYSTEM_PROMPT),
        ("user", "{user_input}")
    ]
)

query_planner_chain = query_planner_prompt | llm.with_structured_output(NewsQuery)

@tool
def retrieve_relevant_news(user_input: str) -> dict:
    """
    Converts a user's news-related question into a structured query object.
    Returns a search_query for Google News and the original user_query for summarization.
    Use this tool FIRST before fetching any news articles.
    """
    result: NewsQuery = query_planner_chain.invoke({"user_input": user_input})
    return {
        "search_query": result.search_query,
        "user_query": result.user_query
    }