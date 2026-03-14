from commons.agents.agent import agent

if __name__ == "__main__":
    query: str = input("Enter your news question: ")
    print("\n🤖 Agent is working...\n")

    result = agent.invoke(
        {"messages": [("user", query)]},
        config={"configurable": {"thread_id": "1"}}
    )

    print("\n" + result["messages"][-1].content)
