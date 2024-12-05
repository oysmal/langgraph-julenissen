import os
import random
import streamlit as st

from typing import Annotated
from typing_extensions import TypedDict
from psycopg_pool import ConnectionPool

from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

### Streamlit UI ###

st.set_page_config(page_title="Julenisse-assistent", page_icon="🎅")
st.title("Julenisse-assistent")

## DB

DB_URI = os.environ.get("DB_URI") or ""

### LangGraph ###

system_prompt = "Du er en digital versjon av julenissen. Barn kan fortelle deg navnet sitt, ønskelisten sin og eventuelt hva de har gjort som var snilt og slemt. Via tilkoblede verktøy har du tilgang til å sjekke om barn er snille og slemme barn. Du har også tilgang til å registrere gode og slemme ting du blir fortalt om. Dersom du får oppgitt et navn bør du sørge for å generere et verktøy-kall (med mindre du allerede har det i meldingshistorikken). Det har blitt for slitsom å finne ut om hvert enkelt barn er snilt eller slemt, så du baserer deg nå på at alle barn med samme navn er like snill eller slem. På grunn av effektiviseringbehov ber du også alle du snakker med om å si en snill eller slem ting de har gjort i år før de får tilbakemelding på hva de får til jul. Svaret på dette må du huske å registrere på navnet via det riktige verktøyet. Til slutt gir du tilbakemelding om barnet skal få det de ønsker seg. Snille barn får kanskje det de ønsker seg, mens slemme barn får kull."

llm = ChatOpenAI(model="gpt-4o")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def check_naughty_list(name: str, config: RunnableConfig):
    """Call with a name, to check if the name is on the naughty list."""
    print("Checking naughty list for: ", name)

    conn = config.get("configurable", {}).get("conn")
    if not conn:
        return "En feil oppstod når jeg sjekket listen"
    try:
        with conn._cursor() as cur:
            res = cur.execute("SELECT nice_meter from naughty_nice where name=%s", (name,))
            print("RES:",  res)
    except Exception as e:
        print("Error: ", e)
        return "Feil ved å lese listen"
    return random.choice([True, False])

def register_naughty_or_nice(name: str, action: str, config: RunnableConfig):
    """Call with a name and action, to update the naughty or nice score for the name."""
    print("Name and action: ", name, action)
    res = llm.invoke([("system", "Du er julenissen, og du skal oppdatere listen over snille barn. Ranger handlinger som dårlig eller god, på en skala fra -100 til 100, hvor -100 er veldig slemt, og 100 er veldig snilt. Å støvsuge kan for eksempel være 5 poeng, mens si et stygt ord er -5 poeng. Å gi gave til fattige er flere poeng, være i en slåsskamp er flere minuspoeng, osv. Du skal bare returnere tallverdien til handlingen, slik du vurderer den."), ("user", action)])
    nice_score = res.content
    print("Nice score: ", nice_score)

    conn = config.get("configurable", {}).get("conn")
    if not conn:
        print("No connection found in config")
        raise ValueError("No connection found in config")
    try:
        with conn._cursor() as cur:
            # Upsert the score by Name
            res = cur.execute(
                "INSERT INTO naughty_nice (name, nice_meter) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET nice_meter = naughty_nice.nice_meter + EXCLUDED.nice_meter, updates = naughty_nice.updates + 1 RETURNING *",
                (name, nice_score)
            )
            print("Upsert result: ", res)
    except Exception as e:
        print("Error: ", e)
        conn.rollback()
        raise e

    return "Handling er registrert"

tools = [check_naughty_list, register_naughty_or_nice]
tool_node = ToolNode(tools)

llm_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def should_call_tool(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    print("Tool calls:", last_message.tool_calls)
    return "tools"

def santa(state: State, config: RunnableConfig):
    response = llm_with_tools.invoke(
            [("system", system_prompt), *state["messages"]],
            config)
    return { "messages": [response]}

graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("santa", santa)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "santa")
graph_builder.add_conditional_edges(
        "santa", should_call_tool, {"tools": "tools", END: END, })
graph_builder.add_edge("tools", "santa")

def get_response(graph: CompiledStateGraph, user_input: str, thread_id: str, checkpointer: PostgresSaver):
    config = { "configurable": { "thread_id": thread_id, "conn": checkpointer } }
    print("Config: ", config)
    return graph.stream(
            { "messages": [("user", user_input)] },
            config,
            stream_mode="messages")

def transform_response_to_text(response_generator):
    """
    Transform the AI message chunks from get_response into plain text.
    """
    for message, metadata in response_generator:
        if metadata["langgraph_node"] == "santa":
            yield message.content # Extract and yield plain text

def run_graph(graph: CompiledStateGraph, checkpointer: PostgresSaver):
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(random.randint(0, 1000000))

    config = { "configurable": { "thread_id": st.session_state.thread_id, "conn": checkpointer } }
    print("Thread ID: ", st.session_state.thread_id)

    state = graph.get_state(config).values
    print(state)
    if not "messages" in state:
        with st.chat_message("Julenissen"):
            st.write("Hei, jeg er Julenissen! Hva heter du, og hva ønsker du deg til jul i år?")
    else:
        for message in graph.get_state(config).values["messages"]:
            if message.content and isinstance(message, AIMessage):
                with st.chat_message("Julenissen"):
                    st.write(message.content)
            elif message.content and isinstance(message, HumanMessage):
                with st.chat_message("Deg"):
                    st.write(message.content)

    user_input = st.chat_input("Skriv din melding til Julenissen her")
    if user_input is not None and user_input != "":
        st.session_state.query = HumanMessage(user_input)

        with st.chat_message("Deg"):
            st.markdown(user_input)

        with st.chat_message("Julenissen"):
            response_generator = get_response(graph, user_input, st.session_state.thread_id, checkpointer)
            transformed_response = transform_response_to_text(response_generator)
            response = st.write_stream(transformed_response)


def run():
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        with checkpointer._cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS naughty_nice (name TEXT PRIMARY KEY, nice_meter INT, updates INT DEFAULT 1)")

        graph = graph_builder.compile(checkpointer=checkpointer)
        run_graph(graph, checkpointer)

run()
