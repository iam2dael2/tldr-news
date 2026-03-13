from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from commons.agents.tool import retrieve_relevant_news
from commons.agents.prompt import AGENT_SYSTEM_PROMPT
from commons.agents.formatter import NewsInsight

from langchain.agents import create_agent
from langchain_groq import ChatGroq

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