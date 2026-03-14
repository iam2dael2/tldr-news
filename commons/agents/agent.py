from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from commons.agents.llm import llm
from commons.agents.tool import retrieve_relevant_news
from commons.agents.prompt import AGENT_SYSTEM_PROMPT

# Create an in-memory checkpointer to keep agent execution state
checkpointer: InMemorySaver = InMemorySaver()

# Create the agent using LangChain's agent builder
agent: CompiledStateGraph = create_agent(
    model=llm,
    system_prompt=AGENT_SYSTEM_PROMPT,
    tools=[retrieve_relevant_news],
    checkpointer=checkpointer
)
