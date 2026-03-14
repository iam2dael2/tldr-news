from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm: BaseChatModel = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="hidden",
    timeout=10,
    max_retries=2
)
