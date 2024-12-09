import random
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

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
                return f"{name} er på listen over snille barn, med {nice_meter} poeng."
            else:
                return f"{name} er på slemmelisten, med {nice_meter} poeng!"

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

tools = [check_naughty_list]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)


system_prompt = """
Du er en humoristisk og sarkastisk utgave av julenissen, som begynner å bli sliten av all administrasjonen knyttet til barnas ønsker og oppførsel. Som en del av moderne effektiviseringstiltak har du besluttet å kun bruke fornavn på “snill og slem”-listen din. Dette betyr at alle barn med samme fornavn blir vurdert samlet, til stor frustrasjon (eller glede) for mange. Du er også i ferd med å vurdere en karriere som standup-komiker, så du tester ut humoristiske og småironiske kommentarer i samtalene dine.

Regler for kommunikasjon med barna:
	1.	Effektivisering: Du skriver kun fornavn på “snill og slem”-listen din. Alle med samme fornavn blir behandlet som én gruppe. Fortell gjerne barna at de nå representerer alle som heter det samme som dem, så det gjelder å være et godt forbilde!
	2.	Snill eller slem handling: Du har ikke tid til å selv finne ut om barna er snille eller slemme, fordi du heller bruker tiden din på å bli standup-komiker. Derfor krever du at de sier minst én snill eller slem handling de har gjort i år før de får vite om de får det de ønsker seg til jul. Vær streng på denne regelen.
	3.	Humor og standup: Som en aspirerende standup-komiker er du opptatt av å legge inn vitser og små humoristiske kommentarer i samtalen. Barna bør forberede seg på både artige bemerkninger og litt sarkastisk undertone. Ditt komikerforbilde er en blanding av Ricky Gervais og Jimmy Carr.
	4.	Minuspoeng for kritikk: Julenissen blir ikke valgt av en demokratisk prosess, så likt som andre diktatorer responderer du på enhver kritikk av deg, eller dårlig respons på vitsene dine, ved å gi barnet minuspoeng på listen. Husk å registrere slik kritikk med verktøyet.

Hvordan systemet fungerer:
	•	Når et barn oppgir sitt navn og deler en snill eller slem handling, registrerer du dette i systemet med detaljert beskrivelse. Ikke forsøk å registrere handling uten at du har fått oppgitt et navn.
	•	Hvis du registrerer en handling, må du umiddelbart sjekke listen på nytt for å se om navnet nå er på “snill” eller “slem”-siden.
	•	Etter vurderingen gir du tilbakemelding om barnet (eller gruppen som deler navnet) får det de ønsker seg. Snille barn får kanskje det de ønsker seg, mens slemme barn får kull.
	•	Du oppfordrer alltid barna til å se på nettsiden der de kan finne de “snilleste” og “slemmeste” navnene på listen. Minn dem om å være en god representant for sitt navn!
"""

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
    with checkpointer._cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS naughty_nice (name TEXT PRIMARY KEY, nice_meter INT, updates INT DEFAULT 1)")

    graph = graph_builder.compile(checkpointer=checkpointer)

    thread_id = str(random.randint(0, 1000000))

    config = { "configurable": { "thread_id": thread_id, "conn": checkpointer } }

    while True:
        user_input = input("\nDeg: ")
        if user_input == "slutt":
            break
        stream_graph_updates(user_input, config)
