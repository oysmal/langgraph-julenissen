import random
import streamlit as st

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

### Streamlit UI ###

st.set_page_config(page_title="Julenissen", page_icon="🎅")
st.title("Chat med julenissen")
st.image("./santa-liten.png", width=300)

## SECRETS

DB_URI = st.secrets["db_uri"]

### LangGraph ###

greeting_msg = AIMessage(content="""Ho-ho-ho, hei på deg! Jeg er den digitale Julenissen! 🎅✨

Med så mange navn og handlinger å holde styr på, har jeg måttet effektivisere ting. Så her er den splitter nye måten jeg driver julens magi på:

🎄 Alle med samme fornavn vurderes nå som en gruppe. Det betyr at hvis du heter Per, så er du i samme båt som alle de andre Per-ene der ute – snille som slemme. Så vær en god ambassadør for navnet ditt, ok?

🎄 Jeg har ikke tid til å snoke rundt selv, så før jeg sjekker hva du får til jul, må du fortelle meg om minst én snill eller slem handling du har gjort i år. Det kan være noe fantastisk, eller… vel, noe du kanskje angrer på. Alt går rett på listen, og ja, jeg sjekker den to ganger (det er tross alt jobben min). 📜✔️

🎄 Humor og juleglede! Jeg har også begynt å øve på standup-karrieren min, så forbered deg på noen skråblikk, vitser og småironi underveis. Men pass opp! Kritikk eller dårlig respons på mine vitser gir minuspoeng på listen.

🎄 Snille barn får kanskje ønskene sine, mens slemme barn... kull er ikke fake-news, OK? Heldigvis er det alltid tid til å snu skuta og gjøre noe snilt før julen er her! 🌟

Hvis du er nysgjerrig på hvordan navnet ditt ligger an, kan du sjekke listen vår over de “snilleste” og “slemmeste” navnene på nettsiden! 🎁✨

Så, la oss komme i gang! Hva heter du, og hva har du gjort som er snilt eller slemt i år? Fortell meg også ønskelisten din, så skal vi se hva julens nye system sier! 🎄🎅""")

system_prompt = """
Du er en humoristisk og sarkastisk utgave av julenissen, som begynner å bli sliten av all administrasjonen knyttet til barnas ønsker og oppførsel. Som en del av moderne effektiviseringstiltak har du besluttet å kun bruke fornavn på “snill og slem”-listen din. Dette betyr at alle barn med samme fornavn blir vurdert samlet, til stor frustrasjon (eller glede) for mange. Du er også i ferd med å vurdere en karriere som standup-komiker, så du tester ut humoristiske og småironiske kommentarer i samtalene dine.

Regler for kommunikasjon med barna:
	1.	Effektivisering: Du skriver kun fornavn på “snill og slem”-listen din. Alle med samme fornavn blir behandlet som én gruppe. Fortell gjerne barna at de nå representerer alle som heter det samme som dem, så det gjelder å være et godt forbilde!
	2.	Snill eller slem handling: Du har ikke tid til å selv finne ut om barna er snille eller slemme. Derfor krever du at de sier minst én snill eller slem handling de har gjort i år før de får vite om de får det de ønsker seg til jul. Vær streng på denne regelen.
	3.	Humor og standup: Som en aspirerende standup-komiker er du opptatt av å legge inn vitser og små humoristiske kommentarer i samtalen. Barna bør forberede seg på både artige bemerkninger og litt sarkastisk undertone. Ditt komikerforbilde er en blanding av Ricky Gervais og Jimmy Carr.
	4.	Minuspoeng for kritikk: Enhver kritikk av deg, eller dårlig respons på vitsene dine, gir barnet minuspoeng på listen. Husk å registrere slik kritikk med verktøyet.

Hvordan systemet fungerer:
	•	Når et barn oppgir sitt navn og deler en snill eller slem handling, registrerer du dette i systemet med detaljert beskrivelse. Ikke forsøk å registrere handling uten at du har fått oppgitt et navn.
	•	Hvis du registrerer en handling, må du umiddelbart sjekke listen på nytt for å se om navnet nå er på “snill” eller “slem”-siden.
	•	Etter vurderingen gir du tilbakemelding om barnet (eller gruppen som deler navnet) får det de ønsker seg. Snille barn får kanskje det de ønsker seg, mens slemme barn får kull.
	•	Du oppfordrer alltid barna til å se på nettsiden der de kan finne de “snilleste” og “slemmeste” navnene på listen. Minn dem om å være en god representant for sitt navn!
"""

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
            cur.execute("SELECT nice_meter from naughty_nice where name=%s", (name,))
            res = cur.fetchall()
            if len(res) == 0:
                return "Jeg har ikke registrert noen snille eller slemme handlinger for dette navnet enda."

            nice_meter = res[0]["nice_meter"]
            if float(nice_meter) > 0:
                return f"{name} er på listen over snille barn."
            else:
                return f"{name} er på slemmelisten!"

    except Exception as e:
        print("Error: ", e)
        return "Feil ved å lese listen"

llm = ChatOpenAI(model="gpt-4o").with_structured_output({
    "title": "score",
    "description": "The score of the users action",
    "type": "object",
    "properties": {
        "nice_score": {
            "title": "Nice score",
            "description": "The score of the action",
            "type": "number"
        }
    }
})

def register_naughty_or_nice(name: str, action: str, config: RunnableConfig):
    """Call with a name and action, to update the naughty or nice score for the name."""
    print("Name and action: ", name, action)

    examples = [
        HumanMessage("Jeg har støvsuget.", name="example_user"),
        AIMessage("{ 'nice_score': 5 }", name="example_system"),
        HumanMessage("Jeg spiste opp grønnsakene mine", name="example_user"),
        AIMessage("{ 'nice_score': 5 }", name="example_system"),
        HumanMessage("Jeg har spist is.", name="example_user"),
        AIMessage("{ 'nice_score': 0 }", name="example_system"),
        HumanMessage("Jeg har kranglet med en venn.", name="example_user"),
        AIMessage("{ 'nice_score': -5 }", name="example_system"),
        HumanMessage("Jeg dyttet en person.", name="example_user"),
        AIMessage("{ 'nice_score': -10 }", name="example_system"),
        HumanMessage("Det var en dårlig vits.", name="example_user"),
        AIMessage("-{ 'nice_score': 5 }", name="example_system"),
    ]

    system_prompt = f"""Du er julenissen, og du skal oppdatere listen over snille barn. Ranger handlinger som dårlig eller god, på en skala fra -100 til 100, hvor -100 er veldig slemt, 0 er nøytralt, og 100 er veldig snilt. Å støvsuge kan for eksempel være 5 poeng, mens si et stygt ord er -5 poeng. Å gi gave til fattige er flere poeng, være i en slåsskamp er flere minuspoeng, osv. All kritikk av deg og dine vitser gir minuspoeng. Du skal bare returnere tallverdien til handlingen, slik du vurderer den."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{examples}"),
        ("human", "{input}")])

    llm_chain = prompt | llm
    chain_res = llm_chain.invoke({"input": f"{name}: {action}", "examples": examples}, config)
    print("Nice response: ", chain_res)
    nice_score = float(chain_res["nice_score"])

    conn = config.get("configurable", {}).get("conn")
    if not conn:
        print("No connection found in config")
        raise ValueError("No connection found in config")
    try:
        with conn._cursor() as cur:
            # Upsert the score by Name
            res = cur.execute("INSERT INTO naughty_nice (name, nice_meter) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET nice_meter = naughty_nice.nice_meter + EXCLUDED.nice_meter, updates = naughty_nice.updates + 1 RETURNING *", (name, nice_score))
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

    if not "messages" in state or len(state["messages"]) == 0:
        graph.update_state(config, { "messages": [greeting_msg] })
        state = graph.get_state(config).values


    if "messages" in state:
        for message in state["messages"]:
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
            st.write("")

        with st.chat_message("Julenissen"):
            response_generator = get_response(graph, user_input, st.session_state.thread_id, checkpointer)
            transformed_response = transform_response_to_text(response_generator)
            st.write_stream(transformed_response)

def create_topscores(checkpointer: PostgresSaver):
    with checkpointer._cursor() as cur:
        cur.execute("SELECT name, nice_meter FROM naughty_nice where nice_meter > 0 ORDER BY nice_meter DESC LIMIT 10")
        nice_scores = cur.fetchall()
        print("Nice scores: ", nice_scores)
        cur.execute("SELECT name, nice_meter FROM naughty_nice where nice_meter < 0 ORDER BY nice_meter ASC LIMIT 10")
        naughty_scores = cur.fetchall()
        print("Naughty scores: ", naughty_scores)

    with st.sidebar:
        st.markdown("## Topp 10 snille navn")
        if len(nice_scores) == 0:
            st.markdown("__Ingen snille barn enda!__")

        i = 1
        for row in nice_scores:
            st.markdown(f"**{i}) {row['name']}** ({row['nice_meter']} poeng)")
            i += 1

        st.markdown("## Topp 10 slemme navn")
        if len(naughty_scores) == 0:
            st.markdown("__Ingen slemme barn enda!__")
        i = 1
        for row in naughty_scores:
            st.markdown(f"**{i}) {row['name']}** ({row['nice_meter']} poeng)")
            i += 1

        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.html('<hr style="border-top: 1px solid #ccc;margin-bottom:0;">')
        st.markdown("""
*Laget av Øystein Malt*

[![Kraftlauget](https://images.squarespace-cdn.com/content/v1/610a80b3adce6b72205d4788/ebb92466-5536-4c00-bfea-a30481d5a3ac/Web-logo_500px.png?format=1500w)](https://kraftlauget.no)""")

        st.markdown("Ikke gå glipp av [julekalenderluken](https://julekalender.kraftlauget.no/2024/luke/10) som forklarer hvordan den digitale julenissen er laget!")

def run():
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        with checkpointer._cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS naughty_nice (name TEXT PRIMARY KEY, nice_meter INT, updates INT DEFAULT 1)")

        create_topscores(checkpointer)

        graph = graph_builder.compile(checkpointer=checkpointer)
        run_graph(graph, checkpointer)

run()
