from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Primary LLM — used for reasoning, summarization, and agent decisions
llm: BaseChatModel = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="hidden",
    timeout=10,
    max_retries=2
)

# Lightweight LLM — used for fast, simple classification tasks (e.g. locale detection)
locale_llm: BaseChatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    timeout=10,
    max_retries=2
)
