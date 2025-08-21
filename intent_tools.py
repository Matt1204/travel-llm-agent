from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.runnables import RunnableConfig
from typing import Annotated, Any
from datetime import datetime
from intent_model import FlightRequirements
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import Annotated, TypedDict, NotRequired
from langchain_core.tools import tool, InjectedToolCallId
from db_connection import db_file
from typing import Optional
import sqlite3
import uuid
import json
from langgraph.graph.message import AnyMessage, add_messages
from graph_setup import PRIMARY_ASSISTANT
from primary_assistant_chain import State

# add this helper near the imports
from amadeus_api import get_amadeus_client

# def _replace_req(old, new):
#     return new
# class IntentElicitationState(AgentState):
#     """Graph state = standard AgentState + custom FlightRequirements."""

#     flight_requirements: Annotated[Optional["FlightRequirements"], _replace_req] = None
#     requirement_id: NotRequired[str] = ""
#     test_counter: NotRequired[int] = 0


@tool
def discover_nearest_airports_for_city(
    city_keyword: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Given a keyword of city or airport, return 1) airport related to the keyword. 2) the top 5 nearest airports around it.
    Use this tool to get information(IATA code) about an airport.
    Parameters:
    city_keyword: str
        A keyword of city or airport. for example: "Toronto", "Billy Bishop".

    Return: a dict with keys: {"airport_found": {...}, "airports_nearby": [...]}
    """
    client = get_amadeus_client()
    # 1) Find airports related to city
    airports = client.airport_search(keyword=city_keyword, limit=50)
    if not airports:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"No airports found for: {city_keyword}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Filter to those in the exact city if we can infer city name match; otherwise keep all
    # Choose 'biggest' by travelers_score desc, fallback by presence
    def _score(a):
        s = a.get("travelers_score")
        return -int(s) if isinstance(s, (int, float)) else 0

    city_airports = sorted(airports, key=_score)
    base_airport = city_airports[0]

    base_lat = base_airport.get("latitude")
    base_lon = base_airport.get("longitude")
    if base_lat is None or base_lon is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Benchmark airport lacks coordinates: {base_airport}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # 2) Nearest airports by flights score, limit 5
    nearest = client.nearest_airport_search(
        latitude=float(base_lat),
        longitude=float(base_lon),
        limit=5,
        radius_km=150,
        # sort="analytics.flights.score",
    )
    nearest_sorted = sorted(nearest, key=_score)
    nearest_sorted = [
        a for a in nearest_sorted if a.get("iata_code") != base_airport.get("iata_code")
    ]

    payload = {
        "city_keyword": city_keyword,
        "airport_found": base_airport,
        "airports_nearby": nearest_sorted,
    }
    return Command(
        update={
            "messages": [
                ToolMessage(content=json.dumps(payload), tool_call_id=tool_call_id)
            ]
        }
    )


def IntentElicitCompleteOrEscalate(
    cancel: bool,
    reason: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """A tool to mark the current task as completed(sync_flight_requirements_to_db called, requirement saved to database) and/or to escalate control of the dialog to the primary assistant if user request is beyond your scope(e.g user abandon current task).
    Do not call this tool if you have not saved the requirements to the database or user request is within your scope, just keep assisting user.
    example 1: "cancel": True, "reason": "User changed their mind about the current task.",
    example 2: "cancel": True, "reason": "I have fully completed the task: <task_description>",
    example 3: "cancel": False, "reason": "I need to search the user's emails or calendar for more information.",
    """
    print(
        "HIIITTTT IntentElicitCompleteOrEscalate called. intent agent => primary assistant"
    )
    return Command(
        goto=PRIMARY_ASSISTANT,
        update={
            "requirement_id": "",
            "flight_requirements": None,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Intent Elicitation Agent hand off to primary assistant. Reason: {reason}",
                )
            ],
            "dialog_state": ["pop"],
        },
    )


@tool
def add_departure_airport_priority(
    priority: int,
    value: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add an departure airport with priority level to the FlightRequirements object. Note: change not applied to database.
    priority: the priority level to add. e.g. 1(highest), 2, 3, 4
    value: the IATA/ICAO code of departure airport, to add. e.g. 'YUL', 'YHU'
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.add_requirement_priority("departure_airport", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Departure airport priority {priority} added: {value}",
                )
            ],
        }
    )


@tool
def add_arrival_airport_priority(
    priority: int,
    value: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add an arrival airport with priority level to the FlightRequirements object. Note: change not applied to database.
    priority: the priority level to add. e.g. 1(highest), 2, 3, 4
    value: the IATA/ICAO code of arrival airport, to add. e.g. 'YYZ', 'YTZ'
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.add_requirement_priority("arrival_airport", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Arrival airport priority {priority} added: {value}",
                )
            ],
        }
    )


@tool
def add_departure_time_priority(
    priority: int,
    value: list[datetime],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add a departure time window with priority level to the FlightRequirements object. Note: change not applied to database.
    priority: the priority level to add. e.g. 1(highest), 2, 3, 4
    value: the departure time window to add. Provide times in ISO-8601 without timezone in the exact schema "YYYY-MM-DDTHH:MM:SS".
    Example: ["2025-08-20T00:00:00", "2025-08-20T23:59:59"].
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.add_requirement_priority("departure_time", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Departure time priority {priority} added: {value}",
                )
            ],
        }
    )


@tool
def add_budget_priority(
    priority: int,
    value: list[int] = [0, 999999],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add a budget boundary with priority level to the FlightRequirements object. Note: change not applied to database.
    priority: the priority level to add. e.g. 1(highest), 2, 3, 4
    value: the lower and upper budget boundary, to add. e.g. [100, 500]. if not explicitly provided, default to [0, 999999].
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.add_requirement_priority("budget", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Budget priority {priority} added: {value}",
                )
            ],
        }
    )


@tool
def update_departure_airport_priority(
    priority: int,
    value: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Update the departure airport at a given priority level in the FlightRequirements object.
    priority: 1 (highest) .. 4
    value: IATA/ICAO code, e.g. 'YUL', 'YHU'
    Note: change not applied to database, only update the FlightRequirements object.
    """
    print("update_departure_airport_priority !!!!!!!!!!!")
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.update_requirement_priority("departure_airport", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Departure airport priority {priority} updated to {value}",
                )
            ],
        }
    )


@tool
def update_arrival_airport_priority(
    priority: int,
    value: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Update the arrival airport at a given priority level in the FlightRequirements object.
    priority: 1 (highest) .. 4
    value: IATA/ICAO code, e.g. 'YYZ', 'YTZ'
    Note: change not applied to database, only update the FlightRequirements object.
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.update_requirement_priority("arrival_airport", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Arrival airport priority {priority} updated to {value}",
                )
            ],
        }
    )


@tool
def update_departure_time_priority(
    priority: int,
    value: list[datetime],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Update the departure time window at a given priority level.
    priority: 1 (highest) .. 4
    value: the departure time window to update. Provide times in ISO-8601 without timezone in the exact schema "YYYY-MM-DDTHH:MM:SS".
    Example: ["2025-08-20T00:00:00", "2025-08-20T23:59:59"].
    Note: change not applied to database, only update the FlightRequirements object.
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.update_requirement_priority("departure_time", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Departure time priority {priority} updated to {value}",
                )
            ],
        }
    )


@tool
def update_budget_priority(
    priority: int,
    value: list[int] = [0, 999999],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Update the budget boundary at a given priority level.
    value must be [lower, upper]. if not explicitly provided, default to [0, 999999].
    Note: change not applied to database, only update the FlightRequirements object.
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.update_requirement_priority("budget", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Budget priority {priority} updated to {value}",
                )
            ],
        }
    )


@tool
def remove_requirement_priority(
    requirement_type: str,
    priority: int,
    *,
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Remove a requirement value with priority level in the FlightRequirements object. Note: change not applied to database.
    requirement_type: the type of requirement to remove. e.g. 'departure_airport', 'arrival_airport', 'departure_time', 'budget', etc.
    priority: the priority level to remove. e.g. 1(highest), 2, 3, 4
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    value = current_req.get_requirement_priority(requirement_type, priority)
    if value is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                        content=f"Requirement {requirement_type} with priority {priority} not found",
                    )
                ],
            }
        )
    current_req.remove_requirement_priority(requirement_type, priority)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                    content=f"Requirement {requirement_type} priority {priority} with value {value} removed",
                )
            ],
        }
    )


@tool
def sync_flight_requirements_to_db(
    requirement_description: str,
    *,
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Used to sync current FlightRequirements object to the database.
    requirement_description: Use concise Natural language to summarize current flight requirements, to help user understand and identify each requirement.
    Note: Only use this tool after user confirms the flight requirements.
    """
    current_req = state.get("flight_requirements", None)
    print(current_req.to_json())
    if current_req is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                        content=f"No flight requirements found. Please add flight requirements first.",
                    )
                ],
            }
        )
    mandatory_fields = ["departure_airport", "arrival_airport", "departure_time"]
    for field in mandatory_fields:
        if current_req.get_requirement_priority(field, 1) is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                            content=f"Mandatory field {field} is missing. Please add it first.",
                        )
                    ],
                }
            )

    arrival_airport_json = json.dumps(
        current_req.get_requirement_obj("arrival_airport").get_all_priorities()
    )

    departure_airport_json = json.dumps(
        current_req.get_requirement_obj("departure_airport").get_all_priorities()
    )

    budget_json = json.dumps(
        current_req.get_requirement_obj("budget").get_all_priorities()
    )

    departure_time = current_req.get_requirement_obj(
        "departure_time"
    ).get_all_priorities()
    departure_time_ISO = {}
    for key, value in departure_time.items():
        departure_time_ISO[key] = [dt.isoformat() for dt in value]
    departure_time_ISO = json.dumps(departure_time_ISO)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cur_req_id = state.get("requirement_id", "")
    requirement_id = None
    if cur_req_id == "":
        # create new requirement
        cursor.execute(
            "INSERT INTO flight_requirements (requirement_id, passenger_id, flight_req_description, departure_airport, arrival_airport, departure_time, budget) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                config.get("configurable", {}).get("passenger_id", None),
                requirement_description,
                departure_airport_json,
                arrival_airport_json,
                departure_time_ISO,
                budget_json,
            ),
        )
    else:
        # update existing requirement
        # make sure record exists
        cursor.execute(
            "SELECT * FROM flight_requirements WHERE requirement_id = ?", (cur_req_id,)
        )
        if cursor.fetchone() is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                            content=f"Error: requirement being updated does not exist in the database.",
                        )
                    ],
                }
            )
        else:
            # update existing requirement
            cursor.execute(
                "UPDATE flight_requirements SET departure_airport = ?, flight_req_description = ?, arrival_airport = ?, departure_time = ?, budget = ? WHERE requirement_id = ?",
                (
                    departure_airport_json,
                    requirement_description,
                    arrival_airport_json,
                    departure_time_ISO,
                    budget_json,
                    cur_req_id,
                ),
            )

    conn.commit()
    cursor.close()
    conn.close()

    return Command(
        update={
            "messages": [
                ToolMessage(
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                    content=f"Flight requirements synced to database",
                )
            ],
        }
    )


"""
CREATE TABLE "flight_requirements" (
  "requirement_id" TEXT,
  "departure_airport" JSON NOT NULL,
  "passenger_id" TEXT NOT NULL,
  "arrival_airport" JSON NOT NULL,
  "departure_time" JSON NOT NULL,
  "budget" JSON,
  PRIMARY KEY ("requirement_id")
);
"""
