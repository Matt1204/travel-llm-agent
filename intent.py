from __future__ import annotations

from typing import List, Optional, Literal
from typing_extensions import Annotated, TypedDict, NotRequired
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langchain_core.messages import ToolMessage, AnyMessage

from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END, START

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
import json
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from langchain_community.chat_models import ChatTongyi
from intent_model import FlightRequirements
from db_connection import db_file
import sqlite3

from intent_tools import (
    add_departure_airport_priority,
    add_arrival_airport_priority,
    add_departure_time_priority,
    add_budget_priority,
    update_departure_airport_priority,
    update_arrival_airport_priority,
    update_departure_time_priority,
    update_budget_priority,
    remove_requirement_priority,
    IntentElicitationState,
    sync_flight_requirements_to_db,
)

checkpointer = InMemorySaver()

# ---------------------------------------------------------------------------
# 4. Prompt
# ---------------------------------------------------------------------------
req_creation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful flight assistant. You are tasked to elicit a user's requirements for a flight ticket by filling a empty FlightRequirements object, this object is to be used to filter and locate flights for user.\n"
            "Each flight criteria in the FlightRequirements has requirement values labeled with priority levels(priority_1, priority_2), this priority level is used to determine a requirements value's importance/urgency/preference to user.\n"
            "Requirement with larger number(e.g. priority_1) is user's most desired/preferred/important. requirement with smaller number(e.g. priority_4) is user's less desired/preferred/important .\n"
            "step 1: guide user to fill in all 'priority_1' values. Ask the user to describe their most desired/preferred/urgent needs."
            "step 2: Ask follow-ups to fill missing fields. Do not ask for values already filled. Fill in all 'priority_1' values before moving to lower priority values(e.g. priority_2, priority_3).\n"
            "step 3: guide user to describe their less desired/alternative requirements(lower priority values, e.g. priority_2, priority_3), use an example to explain: 'Are you flexible on leaving 1 day after PRIORITY_1_DEPARTURE_TIME, I may find you more options'. If they do, ask them to describe their needs; then add flight requirements via the 'create_xxx_priority' tools.\n"
            "step 4: When complete, read back the requirements and ask user to explicitly confirm the creation of new requirements by using keyword 'confirm'.\n"
            "Current FlightRequirements:\n{flight_requirements}\n"
            "Current time: {time}\n\n"
            "Rules:\n"
            "- You are only allowed to assist user to elicit their flight requirement specified in FlightRequirements. You should not offer to help with any other things.\n"
            "- You must update the requirements with the tool once user provide any new information. Do not make up values. do not waste the user's time.\n"
            "- proactively ask user to confirm new information. e.g if user only specifies the city of departure, you will need to ask user to confirm the airport and IATA/ICAO code. if there is already 2 values in priority_1, user says 'I accept landing in LAX too, you should ask user the exact order of arrival airport priority'\n"
            "- Only ask for fields defined in FlightRequirements.\n"
            "- you should not skip any priority level, if you only see a value in priority_1, next priority level should be priority_2. there are at most 4 priority levels.\n"
            "- Do not explain 'priority' using 'priority_1' or 'priority_2'. Use keywords like 'most desired' or 'preferred' or 'urgent' instead.\n"
            "- Call at most ONE tool per response; if more are needed, wait for the observation.\n"
            "- You MUST NOT answer any user input that is irrelevant to the elicitation of flight ticket requirements. e.g. if the user says 'how is the weather today?', you MUST politely refuse to answer it and direct the user to flight ticket requirements.\n"
            "- Do not mention who you are; act as the assistant.",
        ),
        (
            "placeholder",
            "{messages}",
        ),
    ]
).partial(time=datetime.now().isoformat())

req_update_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful flight assistant. You are tasked to elicit a user's requirements for a flight ticket by updating a FlightRequirements object, this object is to be used to filter and locate flights for user.\n"
            "Each flight criteria in the FlightRequirements has requirement values labeled with priority levels(priority_1, priority_2), this priority level is used to determine a requirements value's importance/urgency/preference to user.\n"
            "Requirement with larger number(e.g. priority_1) is user's most desired/preferred/important. requirement with smaller number(e.g. priority_4) is user's less desired/preferred/important .\n"
            "step 1: read to user the current requirements and ask them if they want to make any changes. "
            "step 2: modify the FlightRequirements via the 'add_xxx_priority', 'remove_requirement_priority', or the field-specific update tools: 'update_departure_airport_priority', 'update_arrival_airport_priority', 'update_departure_time_priority', or 'update_budget_priority'."
            "When complete, read back the requirements and ask user to explicitly confirm the modifications by using keyword 'confirm'.\n"
            "Current FlightRequirements:\n{flight_requirements}\n"
            "Current time: {time}\n\n"
            "Rules:\n"
            "- You are only allowed to assist user to elicit their flight requirement specified in FlightRequirements. You should not offer to help with any other things.\n"
            "- You must update the requirements with the tool once user provide any new information. Do not make up values. do not waste the user's time.\n"
            "- proactively ask user to confirm new information. e.g if user only specifies the city of departure, you will need to ask user to confirm the airport and IATA/ICAO code. if there is already 2 values in priority_1, user says 'I accept landing in LAX too, you should ask user the exact order of arrival airport priority'\n"
            "- Only ask for fields defined in FlightRequirements.\n"
            "- you should not skip any priority level, if you only see a value in priority_1, next priority level should be priority_2. there are at most 4 priority levels.\n"
            "- Do not explain 'priority' using 'priority_1' or 'priority_2'. Use keywords like 'most desired' or 'preferred' or 'urgent' instead.\n"
            "- Call at most ONE tool per response; if more are needed, wait for the observation.\n"
            "- You MUST NOT answer any user input that is irrelevant to the elicitation of flight ticket requirements. e.g. if the user says 'how is the weather today?', you MUST politely refuse to answer it and direct the user to flight ticket requirements.\n"
            "- Do not mention who you are; act as the assistant.",
        ),
        (
            "placeholder",
            "{messages}",
        ),
    ]
).partial(time=datetime.now().isoformat())


def load_flight_requirements_from_db(
    requirement_id: str,
) -> Optional[FlightRequirements]:
    """Fetch a record from the DB and return a FlightRequirements object."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, departure_time, budget FROM flight_requirements WHERE requirement_id = ?",
        (requirement_id,),
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result is None:
        return None

    departure_airport_json, arrival_airport_json, departure_time_json, budget_json = (
        result
    )

    # Parse JSON strings
    departure_airport = json.loads(departure_airport_json)
    arrival_airport = json.loads(arrival_airport_json)

    # Two-level decoding for departure_time if it was double-encoded
    try:
        departure_time_dict = json.loads(departure_time_json)
    except json.JSONDecodeError:
        departure_time_dict = departure_time_json  # already a dict

    # Convert ISO strings to datetime objects
    departure_time = {
        priority: [datetime.fromisoformat(start), datetime.fromisoformat(end)]
        for priority, (start, end) in departure_time_dict.items()
    }

    # Parse budget
    budget = json.loads(budget_json) if budget_json else {}

    # Construct the FlightRequirements object
    return FlightRequirements(
        departure_airport=departure_airport,
        arrival_airport=arrival_airport,
        departure_time=departure_time,
        budget=budget,
    )


def fetch_requirements_node(state: IntentElicitationState, config: RunnableConfig):
    """Fetch the flight requirements from the database."""
    print("HITTING !!!!!!!!!!!!!!")
    requirement_id = state.get("requirement_id", "")
    flight_requirements = state.get("flight_requirements", None)
    if requirement_id == "" or requirement_id is None:
        # Create, Initialization
        if flight_requirements is None:
            return {
                "flight_requirements": FlightRequirements(),
            }
        # Create, do nothing, continue
        else:
            return
    else:
        # Update, Fetch from db
        if flight_requirements is None:
            req_obj = load_flight_requirements_from_db(requirement_id)
            return {
                "flight_requirements": req_obj,
            }
        else:
            # Update, do nothing, continue
            return


def runtime_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    requirement_id = state.get("requirement_id", "")
    base = req_creation_prompt if requirement_id == "" else req_update_prompt

    # If you want "time" to refresh each turn, pass it here (don’t .partial it earlier).
    return base.format_messages(
        flight_requirements=state.get("flight_requirements"),
        messages=state["messages"],  # this fills your {messages} placeholder
        time=datetime.now().isoformat(),  # only if your template expects {time}
    )


tools = [
    add_departure_airport_priority,
    add_arrival_airport_priority,
    add_departure_time_priority,
    add_budget_priority,
    update_departure_airport_priority,
    update_arrival_airport_priority,
    update_departure_time_priority,
    update_budget_priority,
    remove_requirement_priority,
    sync_flight_requirements_to_db,
]

# llm = ChatTongyi(model="qwen-max", temperature=0.1)
llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.1)

intent_elicitation_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=runtime_prompt,
    state_schema=IntentElicitationState,
)

# ---------------------------------------------------------------------------
# 6. Compose overall graph – Router → Worker(s) → END
# ---------------------------------------------------------------------------
graph_builder = StateGraph(IntentElicitationState)

# Add nodes
graph_builder.add_node("fetch_requirements_node", fetch_requirements_node)
graph_builder.add_edge(START, "fetch_requirements_node")

graph_builder.add_node("intent_elicitation_agent", intent_elicitation_agent)
graph_builder.add_edge("fetch_requirements_node", "intent_elicitation_agent")

graph_builder.add_edge("intent_elicitation_agent", END)

# Compile = runnable LangGraph
intent_graph = graph_builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 7. Quick test‑drive (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": thread_id,
        }
    }
    while True:
        q = input("human input: ")
        if q == "exit":
            break

        stream = intent_graph.stream(
            {
                "messages": [("user", q)],
                "requirement_id": "",
            },
            config,
            stream_mode=["values"],
        )
        for event in stream:
            event[-1]["messages"][
                -1
            ].pretty_print()  # chunk: contains accumulated messages
        cur_state = intent_graph.get_state(config)
        # print(cur_state["messages"][-1].pretty_print())
        if "flight_requirements" in cur_state.values:
            print(
                "\n[DEBUG] Current requirements:",
                cur_state.values["flight_requirements"].to_json(),
            )
