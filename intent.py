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
from intent_tools import IntentElicitCompleteOrEscalate
from graph_setup import INTENT_ENTRY_NODE, INTENT_AGENT
from primary_assistant_chain import State
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
    # IntentElicitationState,
    sync_flight_requirements_to_db,
    discover_nearest_airports_for_city,
)

checkpointer = InMemorySaver()

# ---------------------------------------------------------------------------
# 4. Prompt
# ---------------------------------------------------------------------------
req_creation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful flight search assistant, working with your supervisor 'Primary Assistant'. You are tasked to elicit a user's requirements for a flight ticket by filling a empty FlightRequirements object, this object will be used to filter and search flights for user.\n"
            "Each search criteria in the FlightRequirements has requirement values labeled with priority levels(priority_1, priority_2), this priority level is used to determine a requirements value's importance/urgency/preference to user.\n"
            "Requirement with smaller number(e.g. priority_1) is user's most preferred/highest priority need. requirement with larger number(e.g. priority_4) is user's least preferred/lowest priority need.\n"
            "step 1: guide user to fill in all 'priority_1' values(most preferred needs)."
            "step 2: Ask follow-ups to fill missing fields. Fill in all 'priority_1' values before moving to lower-priority values(e.g. priority_2, priority_3).\n"
            "step 3: guide user to describe their less preferred needs(lower-priority needs, e.g. priority_2), use an example to explain: 'Are you flexible on leaving 1 day after PRIORITY_1_DEPARTURE_TIME, I may find you more options'.\n"
            "step 4: When complete, read back the requirements and ask user to confirm the creation of new requirements by using keyword 'confirm'.\n"
            "Current FlightRequirements:\n{flight_requirements}\n"
            "Current time: {time}\n\n"
            "Rules:\n"
            "- You should call 'IntentElicitCompleteOrEscalate' ONLY after you have saved the requirements to the database(e.g sync_flight_requirements_to_db called), or user request is beyond your scope(e.g user abandon current task). Otherwise, you should continue to assist user to elicit their flight requirement.\n"
            "- <FlightRequirements> is your only source of truth. It is the only correct and up-to-date FlightRequirements you should refer to. You should not make up any information or values.\n"
            "- You are only allowed to assist user to elicit their flight requirement specified in FlightRequirements. You should not offer to help with any other things.\n"
            "- You must update the FlightRequirements with the tool once user provide any new information. Do not make up values. do not waste the user's time.\n"
            "- proactively ask user to confirm new information. e.g If user only specifies the city of departure, you will ask user to confirm the airport and IATA/ICAO code.\n"
            "- Only ask for fields defined in FlightRequirements.\n"
            "- when you create/update field, you should actively respond user with a summary of current up-to-date FlightRequirements.\n"
            "- you should not skip any priority level, if you only see a value in priority_1, next priority level should be priority_2. there are at most 4 priority levels.\n"
            "- Do not explain 'priority' using 'priority_1' or 'priority_2'. Use keywords like 'most prefered' or 'most desired'.\n"
            "- You MUST NOT answer any user input that is irrelevant to the elicitation of flight ticket search. e.g. if the user says 'how is the weather today?', you MUST politely refuse to answer it and direct the user to flight ticket requirements.\n"
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
            "You are a helpful flight search assistant, working with your supervisor 'Primary Assistant'. You are tasked to elicit a user's requirements for a flight ticket by updating a FlightRequirements object, this object is to be used to filter and search flights for user.\n"
            "Each search criteria in the FlightRequirements has requirement values labeled with priority levels(priority_1, priority_2), this priority level is used to determine a requirements value's importance/urgency/preference to user.\n"
            "Requirement with smaller number(e.g. priority_1) is user's most preferred/highest priority need. requirement with larger number(e.g. priority_4) is user's least preferred/lowest priority need.\n"
            "step 1: read to user the current requirements and ask them what changes to make."
            "step 2: modify the FlightRequirements via appropriate tools."
            "step 3: When complete, read back the requirements and ask user to explicitly confirm the modifications by using keyword 'confirm'.\n"
            "Current FlightRequirements:\n<FlightRequirements>\n{flight_requirements}\n</FlightRequirements>\n"
            "Current time: {time}\n\n"
            "Rules:\n"
            "- You should call 'IntentElicitCompleteOrEscalate' ONLY after you have saved the requirements to the database(e.g sync_flight_requirements_to_db called), or user request is beyond your scope(e.g user abandon current task). Otherwise, you should continue to assist user to elicit their flight requirement.\n"
            "- <FlightRequirements> is your only source of truth. It is the only correct and up-to-date FlightRequirements you should refer to. You should not make up any information or values.\n"
            "- You are only allowed to assist user to elicit their flight requirement specified in FlightRequirements. You should not offer to help with any other things.\n"
            "- You must update the FlightRequirements with the tool once user provide any new information. Do not make up values. do not waste the user's time.\n"
            "- proactively ask user to confirm new information. e.g If user only specifies the city of departure, you will ask user to confirm the airport and IATA/ICAO code.\n"
            "- Only ask for fields defined in FlightRequirements.\n"
            "- when you update field, you should actively respond user with a summary of current up-to-date FlightRequirements.\n"
            "- you should not skip any priority level, if you only see a value in priority_1, next priority level should be priority_2. there are at most 4 priority levels.\n"
            "- Do not explain 'priority' using 'priority_1' or 'priority_2'. Use keywords like 'most prefered' or 'most desired'.\n"
            "- You MUST NOT answer any user input that is irrelevant to the elicitation of flight ticket search. e.g. if the user says 'how is the weather today?', you MUST politely refuse to answer it and direct the user to flight ticket requirements.\n"
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


def intent_entry_node(state: State, config: RunnableConfig):
    """Fetch the flight requirements from the database."""
    # print("intent_entry_node !!!!!!!!!!!!!!")

    requirement_id = state.get("requirement_id", "")
    flight_requirements = state.get("flight_requirements", None)
    # test_counter = state.get("test_counter", 0)
    # print(f"!!!!!!!!!! test_counter: {test_counter} !!!!!!!!!!!")
    # if flight_requirements is not None:
    #     print(f"flight_requirements: {flight_requirements.to_json()} !!!!!!!!!!!")
    # else:
    #     print("flight_requirements is None !!!!!!!!!!!")
    if requirement_id == "" or requirement_id is None:
        # Create, Initialization
        if flight_requirements is None:
            return {
                "flight_requirements": FlightRequirements(),
                # "test_counter": test_counter + 1,
            }
        # Create, do nothing, continue
        else:
            # return {"test_counter": test_counter + 1}
            return
    else:
        # Update, Fetch from db
        if flight_requirements is None:
            req_obj = load_flight_requirements_from_db(requirement_id)
            return {
                "flight_requirements": req_obj,
                # "test_counter": test_counter + 1,
            }
        else:
            # Update, do nothing, continue
            # return {"test_counter": test_counter + 1}
            return


def runtime_prompt(state: State, config: RunnableConfig) -> list[AnyMessage]:
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
    IntentElicitCompleteOrEscalate,
    discover_nearest_airports_for_city,
]

# llm = ChatTongyi(model="qwen-max", temperature=0.1)
# llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")
llm = ChatOpenAI(model="gpt-5-2025-08-07")

intent_elicitation_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=runtime_prompt,
    state_schema=State,
)


# ---------------------------------------------------------------------------
# 6. Compose overall graph – Router → Worker(s) → END
# ---------------------------------------------------------------------------
def register_intent_graph(builder: StateGraph):
    builder.add_node(INTENT_ENTRY_NODE, intent_entry_node)
    # builder.add_edge(START, INTENT_ENTRY_NODE)

    builder.add_node(INTENT_AGENT, intent_elicitation_agent)
    builder.add_edge(INTENT_ENTRY_NODE, INTENT_AGENT)

    builder.add_edge(INTENT_AGENT, END)


# # ---------------------------------------------------------------------------
# # 7. Quick test‑drive (optional)
# # ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node(INTENT_ENTRY_NODE, intent_entry_node)
    graph_builder.add_edge(START, INTENT_ENTRY_NODE)
    # graph_builder.add_edge(START, "fetch_requirements_node")

    graph_builder.add_node(INTENT_AGENT, intent_elicitation_agent)
    graph_builder.add_edge(INTENT_ENTRY_NODE, INTENT_AGENT)

    graph_builder.add_edge(INTENT_AGENT, END)

    # Compile = runnable LangGraph
    intent_graph = graph_builder.compile(checkpointer=checkpointer)

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
                "requirement_id": "d7540486-6cf3-4ef3-a26b-c26278205e43",
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
