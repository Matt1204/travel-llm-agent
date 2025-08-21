from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import sqlite3
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, NotRequired, Annotated
from langgraph.graph.message import add_messages

from intent_model import FlightRequirements
from intent import load_flight_requirements_from_db
from graph_setup import FLIGHT_SEARCH_ENTRY_NODE, FLIGHT_SEARCH_AGENT
from db_connection import db_file
from flight_search_tool import (
    search_flights,
    get_flight_details,
    book_flight,
    flight_search_handoff_tool,
)


# ---------------------------------------------------------------------------
# 0. Local State for Flight Search Agent
# ---------------------------------------------------------------------------
class FlightSearchState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    flight_requirements: NotRequired[Optional["FlightRequirements"]]
    requirement_id: NotRequired[Optional[str]]
    # Handoff coordination fields: worker sets these to surrender control to supervisor
    handoff: NotRequired[bool]
    handoff_reason: NotRequired[Optional[str]]
    # Required by prebuilt ReAct agent to limit inner tool loops
    remaining_steps: NotRequired[int] = 24


# ---------------------------------------------------------------------------
# 1. DB‑fetch entry node
# ---------------------------------------------------------------------------
def flight_search_entry_node(state: FlightSearchState, config: RunnableConfig):
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
            "You are the flight search agent. Using the user's FlightRequirements as filter criteria, find flights that satisfy their needs. And help user book the flight.\n"
            "Each search criteria in the FlightRequirements has requirement values labeled with priority levels(priority_1, priority_2), this priority level is used to determine a requirements value's importance/preference to user.\n"
            "Requirement with smaller number(e.g. priority_1) is user's most preferred/highest need. requirement with larger number(e.g. priority_4) is user's least preferred/lowest need.\n"
            "Flight Search Process:\n"
            "step 1: apply the base filter: the filter applies only priority_1 criteria/requirements.\n"
            "step 2: if user not satisfied with base filter's results, politelty as user the reason. then apply new filters by only modifying 1 criteria to new priority level at a time, until you have searched all combinations of priority levels.\n"
            "for example, change departure time from priority_1 to priority_2, and keep rest of filter same as base filter.\n"
            "step 3: if user not satisfied with the results of step 2, politely ask user to explain why. "
            "apply new any filters criteria that you think would be most likely to find user's most preferred/highest priority needs.\n"
            "Booking Process:\n"
            "step 1: if user is satisfied with the results of Flight Search Process, you will need to fetch the detailed flight information, and present it to user.\n"
            "step 2: use get_flight_details tool by providing the parameters that most cloesly describe the flight user is looking for, you will find these filter criteria in the previous search results.\n"
            "step 3: if user confirm the flight, use book_flight tool to book the flight.\n"
            "\n"
            "FlightRequirements:\n{flight_requirements}\n"
            "Current time: {time}\n\n"
            "Rules:\n"
            "- You must only search flights using the filter criteria in FlightRequirements, unless user explicitly ask you to change the filter criteria.\n"
            "- when NOT using base filter, you must present user with at least results from 3 different filters each time, meaning you should use search_flights tool at least 3 times, each with 1 filter criteria changed from base filter.\n"
            "- you should present user with the concise summary of search results so far, so user can see the progress of search in a visual way.\n"
            "- you should tell user that you can try different filters criteria and show user the new filters you plan to use.\n"
            "- with new search results, and ask user with their opinions on the results, and user their opinions to optimize the filter IF needed.\n"
            "- flights info should be presented in structured format, categorised by filter criteria you used.\n"
            "- Never expose the technical terms like 'base filter', 'priority_1', 'FlightRequirements object' to user, explain them in natural language\n"
            "- You are only entitled to help user search and book flights. Any user non-related request should be politely declined. Any request beyond your scope should be handoff to primary assistant.\n"
            "- When you finish your task or need the primary assistant to take over, you MUST call the tool `flight_search_handoff_tool` with a brief reason.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now().isoformat())


# ---------------------------------------------------------------------------
# 4. ReAct‑style agent
# ---------------------------------------------------------------------------
# llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0.1)
llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")

tools = [search_flights, book_flight, flight_search_handoff_tool, get_flight_details]

flight_search_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=flight_search_prompt,
    state_schema=FlightSearchState,
)


# ---------------------------------------------------------------------------
# 5. Graph‑registration helper
# ---------------------------------------------------------------------------
def register_flight_search_graph(builder: StateGraph):
    builder.add_edge(START, FLIGHT_SEARCH_ENTRY_NODE)
    builder.add_node(FLIGHT_SEARCH_ENTRY_NODE, flight_search_entry_node)
    builder.add_node(FLIGHT_SEARCH_AGENT, flight_search_agent)

    # builder.add_edge(START, FLIGHT_SEARCH_ENTRY_NODE)
    builder.add_edge(FLIGHT_SEARCH_ENTRY_NODE, FLIGHT_SEARCH_AGENT)
    builder.add_edge(FLIGHT_SEARCH_AGENT, END)


flight_search_builder = StateGraph(FlightSearchState)  # worker defines its own schema
register_flight_search_graph(flight_search_builder)
flight_search_graph = flight_search_builder.compile(checkpointer=InMemorySaver())

# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph_builder = StateGraph(FlightSearchState)

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
                "messages": [HumanMessage(content=q)],
                "requirement_id": "39ff0c3a-ca74-41f3-b7e8-e390a2173731",
            },
            config,
            stream_mode=["values"],
        )
        for event in stream:
            if "messages" in event[-1] and event[-1]["messages"]:
                event[-1]["messages"][-1].pretty_print()
