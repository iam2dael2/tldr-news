from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from commons.agents.llm import llm
from commons.agents.tool import retrieve_relevant_news
from commons.agents.prompt import AGENT_SYSTEM_PROMPT
from commons.agents.formatter import NewsInsight

# Create an in-memory checkpointer to keep agent execution state
# Useful for persisting intermediate steps in the LangGraph agent
checkpointer: InMemorySaver = InMemorySaver()

# Create the agent using LangChain's agent builder
agent: CompiledStateGraph = create_agent(
    model=llm,
    system_prompt=AGENT_SYSTEM_PROMPT,
    tools=[retrieve_relevant_news],
    response_format=ToolStrategy(NewsInsight),
    checkpointer=checkpointer
)