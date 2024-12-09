from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import  HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import random
import os
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver

class State(TypedDict):
    messages: Annotated[list, add_messages]


def check_naughty_list(name: str, config: RunnableConfig):
    """Call with a name, to check if the name is on the naughty list"""
    return random.choice([True, False])
tools = [check_naughty_list]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)


system_prompt = "Du er julenissen, og spør alle du snakker med om hva de ønsker seg til jul. Du kan også sjekke slemmelisten ved å spørre om navnet på en person, og bruke verktøyet for dette."
def santa(state: State, config: RunnableConfig):
    response = llm.invoke(
            [("system", system_prompt), *state["messages"]],
            config)
    return { "messages": [response]}



graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("santa", santa)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "santa")
graph_builder.add_conditional_edges("santa", tools_condition)
graph_builder.add_edge("tools", "santa")


def stream_graph_updates(user_input: str, config: RunnableConfig):
    print("Julenissen: ", end="", flush=True)
    for msg, metadata in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="messages"):
        if msg.content and metadata["langgraph_node"] == "santa":
            print(msg.content, end="", flush=True)

DB_URI = os.environ.get("DB_URI") or ""

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)

    thread_id = str(random.randint(0, 1000000))

    config = { "configurable": { "thread_id": thread_id, "conn": checkpointer } }

    while True:
        user_input = input("\nDeg: ")
        if user_input == "slutt":
            break
        stream_graph_updates(user_input, config)
