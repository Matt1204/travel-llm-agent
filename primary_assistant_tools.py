from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.runnables import RunnableConfig
from typing import Annotated, Any
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import Annotated, TypedDict, NotRequired
from typing import Optional

from typing import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command, Send
from graph_setup import (
    INTENT_ENTRY_NODE,
    FLIGHT_SEARCH_INVOKE_NODE,
    FLIGHT_SEARCH_INVOKE_NODE,
    FD_BASELINE_ENTRY_NODE,
)
from langchain_core.messages import ToolMessage, RemoveMessage
from langchain_core.tools import tool, InjectedToolCallId
import sqlite3
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from db_connection import db_file
import time
import json
from datetime import datetime


# from primary_assistant_chain import State


@tool
def fetch_discovered_deals_by_requirement_id(
    requirement_id: str,
):
    """Fetch the results(discovered deals) of a flight discovery agent associated with the given requirement_id.

    Returns:
    - baseline_flight: The baseline flight that complies with all user requirements
    - deals: List of deals, where each deal contains:
      - altered_filter_val: What filter value was changed from baseline
      - discovery_flights: List of better flights found with the altered filter
    """
    if not requirement_id:
        return "Error: requirement_id is required"

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        # Query for all deals with the given requirement_id where is_better_deal = 1
        cursor.execute(
            """
            SELECT 
                baseline_filter,
                baseline_flight,
                discovery_filter,
                discovery_flights,
                altered_filter_val
            FROM flight_discovery 
            WHERE source_requirement_id = ? AND is_better_deal = 1
            ORDER BY deal_id
        """,
            (requirement_id,),
        )

        rows = cursor.fetchall()

        if not rows:
            return f"No discovered deals found for requirement_id: {requirement_id}"

        # Extract baseline flight (should be the same across all deals for the same requirement_id)
        baseline_flight_json = rows[0][1]  # baseline_flight from first row
        baseline_flight = (
            json.loads(baseline_flight_json) if baseline_flight_json else None
        )

        # Process deals
        deals = []
        for row in rows:
            (
                baseline_filter_json,
                _,
                discovery_filter_json,
                discovery_flights_json,
                altered_filter_val_json,
            ) = row

            # Parse JSON fields
            try:
                if discovery_flights_json:
                    discovery_flight_list = json.loads(discovery_flights_json)
                else:
                    discovery_flight_list = []

                # Remove verbose per-leg data; keep summary fields only
                for _idx, _flight in enumerate(discovery_flight_list):
                    if isinstance(_flight, dict):
                        _flight.pop("segments", None)

                altered_filter_val = (
                    json.loads(altered_filter_val_json)
                    if altered_filter_val_json
                    else {}
                )

                deal = {
                    "altered_filter_val": altered_filter_val,
                    "discovery_flights": discovery_flight_list,
                }
                deals.append(deal)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for a deal: {e}")
                continue

        # Add summary information for better LLM understanding
        deal_summaries = []
        for deal in deals:
            altered_val = deal["altered_filter_val"]
            num_flights = len(deal["discovery_flights"])

            # Extract key changes from altered_filter_val
            changes = []
            if "departure_time_window" in altered_val:
                time_window = altered_val["departure_time_window"]
                if len(time_window) == 2:
                    date_str = time_window[0].split()[0]  # Extract date part
                    changes.append(f"departure date changed to {date_str}")

            for key, value in altered_val.items():
                if key != "departure_time_window":
                    changes.append(f"{key} changed to {value}")

            deal_summary = {
                "changes": changes,
                "num_alternative_flights": num_flights,
                "price_range": None,
            }

            # Calculate price range for discovery flights
            if deal["discovery_flights"]:
                prices = []
                for flight in deal["discovery_flights"]:
                    if "min_amount" in flight:
                        prices.append(flight["min_amount"])
                    elif "fares" in flight and flight["fares"]:
                        prices.append(min(fare["amount"] for fare in flight["fares"]))

                if prices:
                    deal_summary["price_range"] = {
                        "min_price": min(prices),
                        "max_price": max(prices),
                    }

            deal_summaries.append(deal_summary)

        result = {
            "baseline_flight": baseline_flight,
            "deals": deals,
            "total_deals": len(deals),
            "deal_summaries": deal_summaries,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error fetching discovered deals: {str(e)}"
    finally:
        cursor.close()
        conn.close()


@tool
def retrieve_booked_flights(
    requirement_id: Optional[str] = None,
    *,
    config: RunnableConfig,
):
    """
    Retrieve booked flights for the current passenger from the database.
    This tool queries the ticket_flights table to find all booked flights for the current passenger.
    You can optionally filter by a specific requirement_id to get flights booked for a trip(flightrequirement).
    Parameters:
    - requirement_id: Optional string to filter flights by a specific flight requirement ID.
                     If provided, only flights booked for this specific requirement will be returned.
                     If not provided, all booked flights for the passenger will be returned.
    """

    # Get passenger_id from config
    passenger_id = config.get("configurable", {}).get("passenger_id", None)
    if not passenger_id:
        return {
            "error": "No passenger_id found in configuration. Unable to retrieve booked flights.",
            "message": "Cannot retrieve flights without passenger identification.",
        }

    # Build SQL query based on whether requirement_id is provided
    if requirement_id:
        sql = """
            SELECT ticket_no, flight_id, fare_conditions, amount, flight_number, 
                   flight_datetime, flight_req_id, passenger_id
            FROM ticket_flights 
            WHERE passenger_id = ? AND flight_req_id = ?
            ORDER BY flight_datetime ASC
        """
        params = (passenger_id, requirement_id)
        filter_msg = f"for requirement ID '{requirement_id}'"
    else:
        sql = """
            SELECT ticket_no, flight_id, fare_conditions, amount, flight_number, 
                   flight_datetime, flight_req_id, passenger_id
            FROM ticket_flights 
            WHERE passenger_id = ?
            ORDER BY flight_datetime ASC
        """
        params = (passenger_id,)
        filter_msg = "for all requirements"

    # Execute query
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
    except Exception as e:
        return {
            "error": f"Database error while retrieving booked flights: {e}",
            "message": "Failed to retrieve booked flights due to database error.",
        }
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    # Process results
    booked_flights = []
    for row in rows:
        (
            ticket_no,
            flight_id,
            fare_conditions,
            amount,
            flight_number,
            flight_datetime,
            flight_req_id,
            _passenger_id,
        ) = row

        # Parse flight_datetime if it's JSON
        try:
            if flight_datetime and flight_datetime.startswith("["):
                flight_times = json.loads(flight_datetime)
                departure_time = flight_times[0] if len(flight_times) > 0 else None
                arrival_time = flight_times[1] if len(flight_times) > 1 else None
            else:
                departure_time = flight_datetime
                arrival_time = None
        except Exception:
            departure_time = flight_datetime
            arrival_time = None

        booked_flights.append(
            {
                "ticket_no": ticket_no,
                "flight_number": flight_number,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "fare_conditions": fare_conditions,
                "amount": amount,
                "flight_req_id": flight_req_id,
            }
        )

    total_flights = len(booked_flights)

    # Generate appropriate message
    if total_flights == 0:
        if requirement_id:
            message = f"No booked flights found for passenger {passenger_id} with requirement ID '{requirement_id}'."
        else:
            message = f"No booked flights found for passenger {passenger_id}."
    else:
        if requirement_id:
            message = f"Found {total_flights} booked flight(s) for passenger {passenger_id} with requirement ID '{requirement_id}'."
        else:
            message = (
                f"Found {total_flights} booked flight(s) for passenger {passenger_id}."
            )

    return {
        "passenger_id": passenger_id,
        "requirement_id": requirement_id,
        "booked_flights": booked_flights,
        "total_flights": total_flights,
        "message": message,
    }


@tool
def fetch_user_flight_search_requirement(
    *,
    config: RunnableConfig,
) -> str:
    """Fetch logged user's flight search requirements from the database.
    Return a list of flight_search_requirement objects.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT requirement_id, flight_req_description, departure_time FROM flight_requirements WHERE passenger_id = ?",
        (passenger_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if len(rows) == 0:
        return "No flight search requirements found. User can create a new flight search requirement to search for flights."

    # Filter out requirements whose departure_time windows are entirely in the past
    now = datetime.now()
    filtered_rows = []
    for requirement_id, description, departure_time_json in rows:
        if not departure_time_json:
            # If no departure_time stored, skip this requirement
            continue
        # Robustly decode departure_time which may be JSON string or already dict-encoded
        dt_dict = None
        try:
            decoded_once = json.loads(departure_time_json)
            # Handle potential double-encoding
            if isinstance(decoded_once, str):
                try:
                    decoded_twice = json.loads(decoded_once)
                    dt_dict = decoded_twice
                except Exception:
                    dt_dict = decoded_once if isinstance(decoded_once, dict) else None
            else:
                dt_dict = decoded_once
        except Exception:
            dt_dict = (
                departure_time_json if isinstance(departure_time_json, dict) else None
            )
        if not isinstance(dt_dict, dict):
            # Malformed data; skip
            continue
        # Keep the row if ANY priority window has not fully expired (end >= now)
        keep = False
        for _priority, window in dt_dict.items():
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                continue
            start_str, end_str = window
            try:
                # ISO 8601 without timezone assumed local
                end_dt = datetime.fromisoformat(end_str)
            except Exception:
                continue
            if end_dt >= now:
                keep = True
                break
        if keep:
            filtered_rows.append((requirement_id, description))

    if len(filtered_rows) == 0:
        return "No flight search requirements found. User can create a new flight search requirement to search for flights."

    return "\n".join(
        [
            f"requirement_id: {req_id}, description: {desc}"
            for req_id, desc in filtered_rows
        ]
    )


@tool
def handoff_to_search_req_agent(
    requirement_id: str,
    task_description: Annotated[str, "Describe what the next agent should do."],
    state: Annotated[MessagesState, InjectedState],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Hand off to control to 'intent_elicitation_agent' to help user create/update flight_search_requirements.
    requirement_id: the id of the flight search requirement to update, obtained from fetch_user_flight_search_requirement. If empty, create a new flight search requirement.
    task_description: describe the task the intent elicitation assistant should do. e.g. "help user create a new flight search requirement", "help user update existing flight search requirement; update departure airport to be SFO"
    """
    # print(
    #     "-------------------------------- handoff to intent elicitation assistant --------------------------------"
    # )
    # --- Trim message history before hand‑off ---
    original_messages = state.get("messages", [])
    trimmed_messages = trim_messages(
        original_messages,
        max_tokens=1024,
        strategy="last",
        token_counter=count_tokens_approximately,
        start_on="human",
        end_on=("ai"),
    )
    deletes = [
        RemoveMessage(id=m.id) for m in original_messages if m not in trimmed_messages
    ]

    # Prepare the hand‑off marker AFTER trimming
    appended_messages = (
        deletes
        + trimmed_messages
        + [
            ToolMessage(
                tool_call_id=tool_call_id,
                content=(
                    f"Control transferred from primary assistant to intent "
                    f"elicitation assistant, to help user on task: {task_description}"
                ),
            )
        ]
    )

    return Command(
        update={
            "messages": appended_messages,
            "dialog_state": ["in_intent_elicitation_assistant"],
            "requirement_id": requirement_id,
            "flight_requirements": None,
        },
        goto=INTENT_ENTRY_NODE,
    )


@tool
def handoff_to_flight_search_agent(
    requirement_id: str,
    task_description: Annotated[str, "Describe what the next agent should do."],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    A tool to handoff control of the dialog to the flight search assistant, whcih helps user search for flights and book flights.
    requirement_id: the requirement_id of the flight search requirement to use to search for flights, obtained from fetch_user_flight_search_requirement. If empty, create a new flight search requirement.
    task_description: describe the task the flight search assistant should do. e.g. "help user search for flights"
    """
    tool_msg = ToolMessage(
        tool_call_id=tool_call_id,
        content=(
            f"Control transferred from primary assistant to flight search assistant, to help user on task: {task_description}"
        ),
    )
    return Command(
        update={
            "dialog_state": ["in_flight_search_assistant"],
            "requirement_id": requirement_id,
            "messages": [tool_msg],
        },
        goto=FLIGHT_SEARCH_INVOKE_NODE,
    )


@tool
def handoff_to_flight_discovery_agent(
    requirement_id: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    A tool to handoff control of the dialog to the flight discovery assistant, which automatically discovers flights according to given Flight Search Requirement.
    Parameter:
    requirement_id: the requirement_id of the flight search requirement to use to search for flights, obtained from fetch_user_flight_search_requirement. If empty, create a new flight search requirement.
    Rule: must confirm the task with user before calling this tool.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Control transferred from primary assistant to flight discovery agent.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": ["in_flight_discovery_assistant"],
            "requirement_id": requirement_id,
        },
        goto=FD_BASELINE_ENTRY_NODE,
    )

class HandoffToFlightSearchAgent(BaseModel):
    """
    A tool to handoff control of the dialog to the flight search assistant, whcih helps user search for flights and book flights.
    requirement_id: the requirement_id of the flight search requirement to use to search for flights, obtained from fetch_user_flight_search_requirement.
    task_description: describe the task specified by the user concisely, to help flight seach agent understand the task. Do not make up any task information yourself.
    """
    requirement_id: str = Field(
        description="the requirement_id of the flight search requirement to use to search for flights, obtained from fetch_user_flight_search_requirement."
    )
    task_description: str = Field(
        description="describe the task specified by the user to help flight seach agent understand the task. Do not make up any task information yourself."
    )

class HandoffToFlightDiscoveryAgent(BaseModel):
    """A tool to handoff control of the dialog to the flight discovery assistant, which automatically discovers flights according to given Flight Search Requirement.
    Parameter:
    Rule: must confirm the task with user before calling this tool.
    """

    requirement_id: str = Field(
        description="the requirement_id of the flight search requirement to use to search for flights, obtained from fetch_user_flight_search_requirement. If empty, create a new flight search requirement."
    )


class ToFlightSearchAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight search and booking."""

    requirement_id: str = Field(
        description="the requirement_id of the flight search requirement to use to search for flights, obtained from fetch_user_flight_search_requirement. If empty, create a new flight search requirement."
    )
    request: str = Field(
        description="Any necessary followup questions the flight search assistant should clarify before proceeding."
    )


# Treated as a tool, to be invoked by the primary assistant
# which will become part of the AIMessage=>tool_name, args when invoked by the primary assistant
class ToFlightAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="describe the task the flight search assistant should do"
    )
    requirement_id: str = Field(
        description="the id of the flight search requirement to use to search for flights, obtained from fetch_user_flight_search_requirement. If empty, create a new flight search requirement."
    )


class ToTaxiAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle taxi related information."""

    request: str = Field(
        description=(
            "Any necessary followup instructions the taxi assistant needs to accomplish the task."
            "e.g. 'create a new taxi request at TIME, pick up at PICKUP_LOCATION, drop off at DROPOFF_LOCATION'"
        )
    )


class WorkerCompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the primary assistant,
    who can re-route the dialog based on the user's needs.
    You must call this tool when you have completed the current task.
    example 1: "cancel": True, "reason": "User changed their mind about the current task.",
    example 2: "cancel": False, "reason": "I have fully completed the task: <task_description>",
    example 3: "cancel": True, "reason": "I need help to search the user's emails or calendar for more information.",
    """

    cancel: bool = True
    reason: str
