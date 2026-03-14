from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from commons.agents.llm import llm
from commons.agents.tool import refine_search_query, retrieve_relevant_news
from commons.agents.prompt import build_agent_system_prompt
from commons.utils.geolocation import get_language_code

# Detect the user's locale language once at startup for use as the ambiguity fallback
_locale_language: str = get_language_code()

# Create an in-memory checkpointer to keep agent execution state
checkpointer: InMemorySaver = InMemorySaver()

# Create the agent using LangChain's agent builder
agent: CompiledStateGraph = create_agent(
    model=llm,
    system_prompt=build_agent_system_prompt(_locale_language),
    tools=[refine_search_query, retrieve_relevant_news],
    checkpointer=checkpointer
)
