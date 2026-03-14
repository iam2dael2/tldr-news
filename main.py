from langchain_core.messages import RemoveMessage, SystemMessage

from commons.agents.agent import agent
from commons.agents.llm import llm
from commons.utils.logger import logger

# Compact conversation history when it exceeds this many messages
_COMPACT_THRESHOLD = 10


def _compact_if_needed(config: dict) -> None:
    """Summarize old messages into one when history grows too long."""
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    if len(messages) <= _COMPACT_THRESHOLD:
        return

    to_summarize = messages[:-2]  # keep the current exchange intact
    history_text = "\n".join(
        f"{m.type}: {m.content[:400]}" for m in to_summarize
    )
    summary = llm.invoke(
        f"Summarize this conversation concisely, preserving all key facts:\n\n{history_text}"
    ).content

    remove_ops = [RemoveMessage(id=m.id) for m in to_summarize]
    summary_msg = SystemMessage(content=f"[Conversation Summary]\n{summary}")
    agent.update_state(config, {"messages": remove_ops + [summary_msg]})

    logger.info(f"Compaction: {len(to_summarize)} messages condensed into 1 summary")
    print(f"\n💾 Conversation compacted ({len(to_summarize)} messages summarized)\n")


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    print("🤖 TL;DR News — press Enter with no input to quit\n")

    while True:
        query: str = input("Enter your news question: ")

        if not query:
            print("⛔ Stop executing...")
            logger.info("Session ended by user")
            break

        print("\n🤖 Agent is working...\n")
        logger.info(f"User query: '{query}'")

        # Snapshot message count before invoke to detect tool usage
        state_before = agent.get_state(config)
        n_before = len(state_before.values.get("messages", []))

        result = agent.invoke({"messages": [("user", query)]}, config=config)

        # Log whether the agent used the tool or answered from context
        new_messages = result["messages"][n_before:]
        tool_called = any(m.type == "tool" for m in new_messages)
        if tool_called:
            logger.info("Decision: External retrieval triggered (tool called)")
        else:
            logger.info("Decision: Answered from conversation context / general knowledge")

        print("\n" + result["messages"][-1].content + "\n")

        # Compact history if it has grown too long
        _compact_if_needed(config)
