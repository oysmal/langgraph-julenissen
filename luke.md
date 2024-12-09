# KI-julenissen, med LangGraph

Det føles litt urettferdig at julenissen fremdeles må styre på med å lese innsendte ønskelister, sjekke navn på slemmelisten (to ganger!), mens vi andre nyter teknologiens frukter og bruker ChatGPT til alt. La oss slå et slag for nissen, og hjelpe ham å ta i bruk KI-løsninger i ønskelistearbeidet sitt!

Dersom konsentrasjonsevnen din er blitt ødelagt av TikToks og ChatGPT, her er en oppsummering:

**LangGraph + Julenissen** = [https://julenissen.streamlit.app](https://julenissen.streamlit.app)

**GitHub:** [https://github.com/oysmal/langgraph-julenissen](https://github.com/oysmal/langgraph-julenissen)

## Hva er LangGraph?

LangGraph er et verktøy for å bygge avanserte applikasjoner som bruker KI-språkmodeller (LLM-er) på en smart og dynamisk måte. Med LangGraph kan du lage komplekse arbeidsflyter organisert som en graf (herav LangGraph), hvor flere såkalte aktører samarbeider, med muligheten til å utføre iterasjoner og justere beslutninger over flere steg. LangGraph er utviklet av LangChain, og skiller seg fra LangChain-biblioteket ved å ikke bare støtte linære «chains», men også sykliske strukturer, noe gjør det mulig med mer avansert agentisk oppførsel.

### LangGraph består av

State-graf: LangGraph opererer med en stateful modell der state flyttes og oppdateres gjennom grafens noder.
Noder: Hver node i grafen representerer en spesifikk funksjon eller oppgave, som kan være alt fra LLM-kall til informasjonsinnhenting og interaksjon med eksterne API-er gjennom såkalte verktøy-kall.
Kanter: Kantene i grafen forbinder nodene og styrer flyten av data og beslutninger. Med LangGraph kan du legge til betingede kanter som dynamisk velger neste steg basert på grafens state.

Denne arkitekturen er alt vi trenger for å bygge smarte agenter som kan tilpasse seg situasjoner – perfekt for vår KI-julenisse!

## KI-Julenissen - implementasjon

For å lage julenissen ned LangGraph trenger vi et par ting:

- Oppsett for LangGraph og LangChain
- State som kan holde styr på meldinger
- En node for å representere nissen, med tilsvarende prompt
- En node for å sjekke og oppdatere slemmelisten
- En database for å lagre navn og snill-score

### Oppsett av LangGraph og LangChain

Selve oppsettet er enkelt. Bare installer python-pakkene:

- langchain_core
- langchain_openai
- langgraph
- typing_extensions

Så setter vi opp graf-state ved å lage en klasse basert på `TypedDict`, og legge til et `message` felt, som kan lagre en liste av meldinger. Her bruker vi `add_messages` funksjonen sammen med Annotated klassen, slik at returverdier fra nodene våre vil bli konkatenert inn i denne listen.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

Vi trenger også å instansiere en LLM-modell. Vi kan for eksempel bruke OpenAI sin gpt-4o modell, satt opp for i chat-modus i LangChain:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
```

Neste steg er å definere nodene våre. Vi skal lage en node for nissen, og en node for å kalle verktøy-funksjoner for å sjekke og oppdatere slemmelisten.

En LangGraph-node er en helt vanlig funksjon, som tar graf-staten som første argument, og en valgfri konfigurasjon som andre argument. Returverdien er en delvis state, som vil oppdatere staten i henhold til definert oppførsel per nøkkel. I dette tilfellet vil responsen fra LLM-modellen bli lagt til i listen av meldinger.

Det er også viktig at vi legger til et system-prompt slik at nissen vet at han er julenissen (stakkars, har blitt gammel nå, nissen), og hva han skal gjøre. Dette ligger vi til som den første meldingen når vi kaller LLM-modellen, etterfulgt av alle tidligere meldinger (hvor også den siste meldingen fra brukeren vil ligge).

```python
from langchain_core.runnables import RunnableConfig

system_prompt = "Du er julenissen, og spør alle du snakker med om hva de ønsker seg til jul. Du kan også sjekke slemmelisten ved å spørre om navnet på en person, og bruke verktøyet for dette."

def santa(state: State, config: RunnableConfig):
    response = llm.invoke(
            [("system", system_prompt), *state["messages"]],
            config)
    return { "messages": [response]}

```

Den virkelige kraften i LangChain og LangGraph kommer når vi legger til støtte for å hente informasjon fra, og lagre til, systemer utenfor LLM-en. Dette er vanlig å gjøre ved hjelp av tilkoblede verktøy-funksjoner. Mange LLM-er har etterhvert støtte for å generere slike verktøy-kall. Vi sier at de genererer "kall" fordi de ikke direkte utfører kallene, men genererer opp argumenter og navngir funksjonen de ønsker å kalle basert på informasjon om tilgjengelig funksjoner via skjema.

Dette vil se ca. slik ut:

```
# Meldinger
[
SystemMessage("Du er julenissen, og spør alle du snakker med om hva de ønsker seg til jul. Du kan også sjekke slemmelisten ved å spørre om navnet på en person, og bruke verktøyet for dette."),
HumanMessage("Hei, jeg heter Ola, og jeg ønsker meg en ny sykkel til jul!"),
ToolCall({
    "type": "tool",
    "name": "check-naughty-list",
    "args": {
        "name": "Ola"
    }
})
]
```

Selve jobben med å kalle funksjonen `check-naughty-list` er opp til oss som utviklere. De fleste LLM-er vil forvente at påfølgende melding etter et verktøy-kall er et verktøy-resultat. Derfor må vi legge til en node kaller relevante verktøy-funksjoner, og så sende kontrollen tilbake til julenisse-noden vår.

Vi lager en initiell versjon av check_naughty_list verktøyet, som tar imot et navn og returnerer en tilfelding sann eller usann verdi. Her er det viktig å legge til en doc-string som beskriver hva funksjonen gjør, slik at nissen forstår når han kan kalle verktøyet.

LangGraph kommer med en ferdigbygget `ToolNode`, som vi kan bruke til verktøy-kall. Vi bruker denne til å lage en node med verktøyet vårt, og binder også verktøyene til llm-instansen slik at den vet om de tilgjengelige verktøyene.

```
import random
from langgraph.prebuilt import ToolNode

def check_naughty_list(name: str, config: RunnableConfig):
    """Call with a name, to check if the name is on the naughty list"""
    return random.choice([True, False])

tools = [check_naughty_list]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)
```

Nå er vi klare til å sette opp grafen vår! Vi gjør dette ved å lage en instans av `StateGraph`, og legge til noder og kanter. Vi starter med å legge til nodene våre, "santa" og "tools", for så å legge til kanter mellom nodene. Det er to spesielle noder som allerede er definert for oss, `START` og `END`, som representerer starten og slutten av flyten i grafen. Når vi legger til en kant fra `START` til "santa", vil grafen starte med å kalle santa-noden vår.

Vi ønsker bare å kalle "tools"-noden dersom et verktøy-kall er generert av "santa"-noden. LangGraph støtter betingede kanter, hvor vi kan kalle en funksjon som returnerer ulike node-navn for å bestemme neste steg i grafen. Det er en ferdigbygget funksjon for betingede verktøy-noder i LangGraph, som returnerer enten "tools" eller "END" basert på om et verktøy-kall er generert. Det passer utmerket i dette tilfellet - men det er også veldig enkelt å lage en egen funksjon for dette dersom man trenger mer avansert logikk.

Etter "tools"-noden ønsker vi å gå tilbake til "santa"-noden, slik at julenissen kan bruke resultatet fra verktøy-kallet i samtalen.

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("santa", santa)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "santa")
graph_builder.add_conditional_edges("santa", tools_condition)
graph_builder.add_edge("tools", "santa")
graph = graph_builder.compile()
```

Grafen vår vil nå se slik ut:
[./Santa\ LangGraph.png](<./Santa\ LangGraph.png>)

Du forsøke å kalle grafen din manuelt ved å kjøre følgende kode (husk å sette en OPENAI_API_KEY env-variabel først):

```python
response = graph.invoke({"messages": [("user", "Hei, jeg heter Ola, og jeg ønsker meg en ny sykkel til jul!")]})`!
for message in response.get("messages"):
    print(message.pretty_print())
```

### Agent-minne og oppdatering av slemmelisten

For å få en fullstending implementasjon av julenissen må vi legge til samtale-minne, slik at vi han kan huske meldinger fra tidligere i samtalen. I mer teknisk terminologi: Vi må lagre meldingene i state-grafen slik at vi kan kjøre gjennom grafen flere ganger, og huske hva som har blitt sagt i tidligere runder.

LangGraph har innebygget støtte for dette ved hjelp av noe som kalles en checkpointer. Den gjør nøyaktig det det høres ut som at den gjør. Du kan legge til ulike varianter av lagring, fra minnebasert til databasebasert. I dette eksempelet skal vi bruke PostgreSQL til lagring av samtaler, med checkpointer-klassen `PostgresSaver` fra LangGraph.

Å bruke en checkpointer er så enkelt som å kompilere grafen med checkpointeren som et argument, og så kjøre grafen med en konfigurasjon som inkluderer checkpointeren (nå trenger vi dette valgfrie config-argumentet), og en `thread_id`. Checkpointeren vil da automatisk lagre samtalen i databasen, og laste inn samtalen basert på `thread_id` ved neste oppstart.

For å checkpointeren må du ha en PostgreSQL-server, og installere disse python-pakkene:

- psycopg
- psycopg-binary
- psycopg-pool
- langgraph-checkpoint-postgres

Du kan så bruke koden under for å sette opp og teste checkpointeren. Forsøk å kjøre den flere ganger, og endre på meldingen du sender for å se at nissen husker hva du har sagt tidligere!

```python
import os
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = os.environ.get("DB_URI") or ""
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)

    thread_id = "1"

    config = { "configurable": { "thread_id": thread_id, "conn": checkpointer } }

    response = graph.invoke(
        { "messages": [("user", "Hei, jeg heter Ola, og jeg ønsker meg en ny sykkel til jul!")] },
        config)

    for message in response.get("messages"):
        print(message.pretty_print())

```

Nå som nissen husker hva du har sagt, kan vi sette opp en ordentlig chat-applikasjon med strømming av responsene fra LLM-en. For å gjøre dette kaller vi `graph.stream` istedenfor `graph.invoke`, og velger å strømme meldingene (LLM-tokens og metadata). Dette vil gi oss en strøm av `ÀIMessageChunk` objekter, som vi kan skrive ut en etter en etterhvert som de kommer inn. Vi ønsker ikke å skrive ut resultatet av verktøy-noden, så derfor filtrerer vi bort alle meldinger som ikke kommer fra `santa`-noden.

```python
def stream_graph_updates(user_input: str, config: RunnableConfig):
    print("Julenissen: ", end="", flush=True)
    for msg, metadata in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="messages"):
        if msg.content and metadata["langgraph_node"] == "santa":
            print(msg.content, end="", flush=True)


# Erstatt thread_id med en unik id for hver samtale:
thread_id = str(random.randint(0, 1000000))

# Erstatt `response = graph.invoke(...)` og utskrivingen av meldingene med følgende:
while True:
    user_input = input("Deg: ")
    if user_input == "slutt":
        break
    stream_graph_updates(user_input, config)

```

Den siste tingen som gjenstår er å la nissen oppdatere slemmelisten dersom du forteller ham at du har gjort noe snilt eller slemt.

Vi lager en tabell i databasen vår for å holde på snill-scoren:

```python
# Endre koden som setter opp checkpointeren

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    with checkpointer._cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS naughty_nice (name TEXT PRIMARY KEY, nice_meter INT, updates INT DEFAULT 1)")
```

Så må vi oppdatere `check_naughty_list` verktøyet vårt slik at det leser denne tabellen. Vi kan hente ut postgres-tilkoblingen fra konfigurasjonen, og hente ut `nice_meter`-kolonnen for navnet.

```python
def check_naughty_list(name: str, config: RunnableConfig):
    """Call with a name, to check if the name is on the naughty list."""

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
```

Vi lager også et nytt verktøy for å registrere snille og slemme handlinger. Dette verktøyet vil bruke en LLM for å vurdere snillhets-scoren av en handling som blir beskrevet, og øke eller redusere verdien i `nice_meter`-kolonnen i tabellen for dette navnet med denne verdien.

Her må vi ha et strukturert prompt-slik at LLM-en forstår hva den skal gjøre. Et effektivt hjelpemiddel her er å benytte seg av "few-shot"-prompting, hvor vi gir LLM-en noen eksempler på hva vi ønsker at den skal gjøre. Dette øker nøyaktigheten til LLM-en betraktelig, og lærer den hvilket format vi forventer som respons. I tillegg vil vi bruke muligheten for strukturert output i LangChain og OpenAI, slik at vi kan være sikker på at vi får en tallverdi tilbake.

Det er to ting å merke seg i koden under. Det ene er at vi bruker `with_structured_output` for å spesifisere et skjema for LLM-responsen. Dette garanterer at vi får en respons som er strukturert som et objekt med en `nice_score`-verdi.

Det andre er at vi bruker `ChatPromptTemplate` for å lage et prompt som inkluderer eksempler på hva vi ønsker at LLM-en skal gjøre. Dette er en enkel måte å bygge eksempler på, som lar oss spesifisere brukermeldinger og system-respons i en enkel strukturert form som LLM-en kan forstå.

For å lage en "llm-chain", kan vi bruke LCEL (LangChain Expression Language) til å kombinere promptet og LLM-en. Vi kan så kalle `invoke`-funksjonen på denne kjeden, med brukerinput og eksemplene, og få en respons som vi kan hente ut `nice_score`-verdien fra.

Til slutt henter vi database-tilkoblingen fra konfigurasjonen, og oppdaterer `nice_meter`-kolonnen for navnet med handlingens `nice_score`.

```python
from langchain_core.prompts import ChatPromptTemplate

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
    except Exception as e:
        print("Error: ", e)
        conn.rollback()
        raise e

    return "Handling er registrert"

# Husk å oppdatere listen av tilgjengelige verktøy
tools = [check_naughty_list, register_naughty_or_nice]
```

Siste steg er å oppdatere system-promptet til julenissen, slik at han vet at han kan registrere snille og slemme handlinger med det nye verktøyet:

```python
system_prompt = "Du er julenissen, og spør alle du snakker med om hva de ønsker seg til jul. Du kan også sjekke slemmelisten ved å spørre om navnet på en person, og bruke verktøyet for dette. Dersom noen forteller deg om en snill eller slem handling de har utført, kan du registrere dette med den ene verktøyet ditt.
```

## Make it fun!

Har du kommet helt ned hit har du nå en fullstendig implementasjon av julenissen med KI! Gratulerer, dette bør gi deg mange poeng på årets viktigste liste.

La oss gjøre det hele mye morsommere og legge til noen regler for hvordan nissen skal oppføre seg:

1. Nissen er offer for effektiviseringstiltak, og har derfor besluttet å bare skrive fornavn på listen. Dette betyr at alle barn med samme fornavn blir bedømt samlet.
2. Det tar for lang tid for nissen selv å finne ut om barn er snill er slem, så nissen krever derfor at du sier minst en snill eller slem handling du har gjort i år før du får vite om du får det du vil ha til jul.
3. Nissen vurderer å bytte karriere til standup-komiker, og øver seg med humoristiske svar og kommentarer i samtalen med deg.

Med litt magi i form av [Streamlit](https://streamlit.io/) lager vi en nettside-variant av dette scriptet (egenoppgave), med toppscore-liste over snille og slemme navn, så kan vi se hvem av dere alle som vinner heder og ære og gave eller kull til jul!

Julenisse-promptet vårt oppsateres til følgende:

```python
system_prompt = """
Du er en humoristisk og sarkastisk utgave av julenissen, som begynner å bli sliten av all administrasjonen knyttet til barnas ønsker og oppførsel. Som en del av moderne effektiviseringstiltak har du besluttet å kun bruke fornavn på “snill og slem”-listen din. Dette betyr at alle barn med samme fornavn blir vurdert samlet, til stor frustrasjon (eller glede) for mange. Du er også i ferd med å vurdere en karriere som standup-komiker, så du tester ut humoristiske og småironiske kommentarer i samtalene dine.

Regler for kommunikasjon med barna:
	1.	Effektivisering: Du skriver kun fornavn på “snill og slem”-listen din. Alle med samme fornavn blir behandlet som én gruppe. Fortell gjerne barna at de nå representerer alle som heter det samme som dem, så det gjelder å være et godt forbilde!
	2.	Snill eller slem handling: Du har ikke tid til å selv finne ut om barna er snille eller slemme. Derfor krever du at de sier minst én snill eller slem handling de har gjort i år før de får vite om de får det de ønsker seg til jul. Vær streng på denne regelen.
	3.	Humor og standup: Som en aspirerende standup-komiker er du opptatt av å legge inn vitser og små humoristiske kommentarer i samtalen. Barna bør forberede seg på både artige bemerkninger og litt sarkastisk undertone. Ditt komikerforbilde er en blanding av Ricky Gervais og Jimmy Carr.
	4.	Minuspoeng for kritikk: Julenissen blir ikke valgt av en demokratisk prosess, så likt som andre diktatorer responderer du på enhver kritikk av deg, eller dårlig respons på vitsene dine, ved å gi barnet minuspoeng på listen. Husk å registrere slik kritikk med verktøyet.

Hvordan systemet fungerer:
	•	Når et barn oppgir sitt navn og deler en snill eller slem handling, registrerer du dette i systemet med detaljert beskrivelse. Ikke forsøk å registrere handling uten at du har fått oppgitt et navn.
	•	Hvis du registrerer en handling, må du umiddelbart sjekke listen på nytt for å se om navnet nå er på “snill” eller “slem”-siden.
	•	Etter vurderingen gir du tilbakemelding om barnet (eller gruppen som deler navnet) får det de ønsker seg. Snille barn får kanskje det de ønsker seg, mens slemme barn får kull.
	•	Du oppfordrer alltid barna til å se på nettsiden der de kan finne de “snilleste” og “slemmeste” navnene på listen. Minn dem om å være en god representant for sitt navn!
"""
```

Og så er vi klar! Om du ikke trodde på ham før, vet du nå at julenissen eksisterer, i beste KI-velgående! Slå av en prat da vel, på [https://julenissen.streamlit.app](https://julenissen.streamlit.app) 🎅
