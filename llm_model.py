from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, MessagesState
from dotenv import  load_dotenv

from retriever import Retriever
from prompts import (
        get_code_index_agent_messages,
        get_documentation_agent_messages,
        get_qa_agent_messages,
        get_history_agent_messages
)
from typing import Literal

load_dotenv()
retriever = Retriever()

llm = ChatGroq(model="llama-3.3-70b-versatile")


def router(state: MessagesState) -> str:
    """ Determines the next node to call based on the user's question. """
    # It's better practice to get the content from the last message
    user_question = state["messages"][-1].content.lower()

    if "code" in user_question or "function" in user_question:
        return "code_index_agent"
    elif "documentation" in user_question or "docs" in user_question:
        return "documentation_agent"
    elif "history" in user_question or "commit" in user_question:
        return "history_agent"
    else:
        return "qa_agent"


def code_index_agent(state: MessagesState):
    user_question = state["messages"][-1].content
    context =  retriever.retrieve(user_question)
    print("code agent", context)
    messages = get_code_index_agent_messages(user_question, context)
    reply = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": reply.content}], "answer": reply.content}


def documentation_agent(state: MessagesState):
    user_question = state["messages"][-1].content
    context = retriever.retrieve(user_question)
    print("doc agent", context)
    messages = get_documentation_agent_messages(user_question, context)
    reply = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": reply.content}], "answer": reply.content}

def qa_agent(state: MessagesState):
    user_question = state["messages"][-1].content
    context = retriever.retrieve(user_question)
    print("qa agent", context)
    messages = get_qa_agent_messages(user_question, context)
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}],"answer": reply.content}

def history_agent(state: MessagesState):
    user_question = state["messages"][-1].content
    context = retriever.retrieve(user_question)
    print("history agent",context)
    messages = get_history_agent_messages(user_question, context)
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}], "answer": reply.content}

def agent_build():
    builder = StateGraph(MessagesState)

    # Add all the nodes
    builder.add_node("code_index_agent", code_index_agent)
    builder.add_node("documentation_agent", documentation_agent)
    builder.add_node("qa_agent", qa_agent)
    builder.add_node("history_agent", history_agent)

    # The entry point is now a conditional one based on the router function
    builder.set_conditional_entry_point(
        router,
        {
            "code_index_agent": "code_index_agent",
            "documentation_agent": "documentation_agent",
            "qa_agent": "qa_agent",
            "history_agent": "history_agent",
        },
    )

    # Add edges from each agent to the end
    builder.add_edge("code_index_agent", END)
    builder.add_edge("documentation_agent", END)
    builder.add_edge("qa_agent", END)
    builder.add_edge("history_agent", END)

    graph = builder.compile()
    return graph


def main():
    graph = agent_build()
    print("Chat with your Git repository! Type 'exit' to end the session.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        # Pass input as a message to conform to MessagesState
        response = graph.invoke({"messages": [("user", user_input)]})
        answer=  response["messages"][-1].content if hasattr(response["messages"][-1], "content") else response["messages"][-1]["content"]
        print(f"Bot: {answer}")

if __name__ == "__main__":
    main()