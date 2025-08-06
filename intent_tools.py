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


# add this helper near the imports
def _replace_req(old, new):
    return new


class IntentElicitationState(AgentState):
    """Graph state = standard AgentState + custom FlightRequirements."""

    # flight_requirements: NotRequired["FlightRequirements"] = None
    flight_requirements: Annotated[Optional["FlightRequirements"], _replace_req] = None
    requirement_id: NotRequired[str] = ""


@tool
def add_departure_airport_priority(
    priority: int,
    value: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[IntentElicitationState, InjectedState],
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
    state: Annotated[IntentElicitationState, InjectedState],
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
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add a departure time window with priority level to the FlightRequirements object. Note: change not applied to database.
    priority: the priority level to add. e.g. 1(highest), 2, 3, 4
    value: the departure time window, to add. e.g. [datetime(2025, 8, 1, 10, 0), datetime(2025, 8, 2, 20, 0)]
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
    value: list[int],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add a budget boundary with priority level to the FlightRequirements object. Note: change not applied to database.
    priority: the priority level to add. e.g. 1(highest), 2, 3, 4
    value: the lower and upper budget boundary, to add. e.g. [100, 500]
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
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Update the departure airport at a given priority level in the FlightRequirements object.
    priority: 1 (highest) .. 4
    value: IATA/ICAO code, e.g. 'YUL', 'YHU'
    Note: change not applied to database, only update the FlightRequirements object.
    """
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
    state: Annotated[IntentElicitationState, InjectedState],
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
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Update the departure time window at a given priority level.
    value must be a list of two datetimes: [start, end].
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
    value: list[int],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Update the budget boundary at a given priority level.
    value must be [lower, upper].
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
    state: Annotated[IntentElicitationState, InjectedState],
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
    state: Annotated[IntentElicitationState, InjectedState],
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
                "UPDATE flight_requirements SET departure_airport = ?, arrival_airport = ?, departure_time = ?, budget = ? WHERE requirement_id = ?",
                (
                    departure_airport_json,
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
