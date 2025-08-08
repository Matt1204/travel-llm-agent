from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import sqlite3
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from intent_model import FlightRequirements
from intent import load_flight_requirements_from_db
from primary_assistant_chain import State
from graph_setup import FLIGHT_SEARCH_ENTRY_NODE, FLIGHT_SEARCH_AGENT
from db_connection import db_file
from flight_search_tool import search_flights, get_flight_info_by_id, book_flight


# ---------------------------------------------------------------------------
# 1. DB‑fetch entry node
# ---------------------------------------------------------------------------
def flight_search_entry_node(state: State, config: RunnableConfig):
    """Load user's FlightRequirements from DB using `requirement_id` in state."""
    requirement_id: str | None = state.get("requirement_id", None)
    if not requirement_id:
        # No requirement id supplied – nothing to do (let supervisor handle).
        return

    req_obj = load_flight_requirements_from_db(requirement_id)
    # The calling graph will merge this into overall state.
    return {"flight_requirements": req_obj}


# ---------------------------------------------------------------------------
# 3. Prompt
# ---------------------------------------------------------------------------
flight_search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a flight‑search assistant. Using the user's FlightRequirements as filter criteria, "
            "find flights that satisfy their needs. And help user book the flight.\n\n"
            "Each search criteria in the FlightRequirements has requirement values labeled with priority levels(priority_1, priority_2), this priority level is used to determine a requirements value's importance/preference to user.\n"
            "Requirement with smaller number(e.g. priority_1) is user's most preferred/highest priority need. requirement with larger number(e.g. priority_4) is user's least preferred/lowest priority need.\n"
            "Flight Search Process:\n"
            "step 1: apply the base filter: the filter applies only priority_1 criteria/requirements.\n"
            "step 2: if user not satisfied with base filter's results, politelty as user the reason. then apply new filters by only modifying 1 criteria to new priority level at a time, until you have searched all combinations of priority levels.\n"
            "for example, change departure time from priority_1 to priority_2, and keep rest of filter same as base filter.\n"
            "step 3: if user not satisfied with the results of step 2, politely ask user to explain why. "
            "apply new any filters criteria that you think would be most likely to find user's most preferred/highest priority needs.\n"
            "Booking Process:\n"
            "step 1: if user is satisfied with the results of Flight Search Process.\n"
            "step 2: use get_flight_info_by_id tool to get the detailed flight information, and present it to user.\n"
            "step 3: if user confirm the flight, use book_flight tool to book the flight.\n"
            "\n"
            "FlightRequirements:\n{flight_requirements}\n"
            "Current time: {time}\n\n"
            "Rules:\n"
            "- You must only search flights using the filter criteria in FlightRequirements, unless user explicitly ask you to change the filter criteria.\n"
            "- you should actively present user with the search results so far, so user can see the progress of search in a visual way.\n"
            "- you should actively tell user that you can try different filters criteria and show user the new filters you plan to use.\n"
            "- with new search results, and ask user with their opinions on the results, and user their opinions to optimize the filter IF needed.\n"
            "- when NOT using base filter, you must present user with at least results from 3 different filters each time, meaning you should use search_flights tool at least 3 times, each with 1 filter criteria changed.\n"
            "- flights info should be presented in structured format, categorised by filter criteria you used.\n"
            "- Do *not* edit the FlightRequirements here – that is handled by intent_elicitation_agent.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now().isoformat())


# ---------------------------------------------------------------------------
# 4. ReAct‑style agent
# ---------------------------------------------------------------------------
# llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0.1)
llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")

tools = [search_flights, get_flight_info_by_id, book_flight]

flight_search_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=flight_search_prompt,
    state_schema=State,
)


# ---------------------------------------------------------------------------
# 5. Graph‑registration helper
# ---------------------------------------------------------------------------
def register_flight_search_graph(builder: StateGraph):
    builder.add_node(FLIGHT_SEARCH_ENTRY_NODE, flight_search_entry_node)
    builder.add_node(FLIGHT_SEARCH_AGENT, flight_search_agent)

    builder.add_edge(START, FLIGHT_SEARCH_ENTRY_NODE)
    builder.add_edge(FLIGHT_SEARCH_ENTRY_NODE, FLIGHT_SEARCH_AGENT)
    builder.add_edge(FLIGHT_SEARCH_AGENT, END)


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph_builder = StateGraph(State)

    register_flight_search_graph(graph_builder)
    # Compile = runnable LangGraph
    checkpointer = InMemorySaver()
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
                "requirement_id": "73e020b5-0a2f-4320-92c0-24de6fa3fd97",
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
