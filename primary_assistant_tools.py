from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.runnables import RunnableConfig
from typing import Annotated, Any
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import Annotated, TypedDict, NotRequired

from typing import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command, Send
from graph_setup import INTENT_ENTRY_NODE, FLIGHT_SEARCH_INVOKE_NODE
from langchain_core.messages import ToolMessage, RemoveMessage
from langchain_core.tools import tool, InjectedToolCallId
import sqlite3
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from db_connection import db_file
import time

# from primary_assistant_chain import State


@tool
def fetch_user_flight_search_requirement(
    *,
    config: RunnableConfig,
) -> str:
    """Fetch logged user's flight search requirements from the database.
    Return a list of flight search requirements.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT requirement_id, flight_req_description FROM flight_requirements WHERE passenger_id = ?",
        (passenger_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if len(rows) == 0:
        return "No flight search requirements found. User can create a new flight search requirement to search for flights."
    return "\n".join(
        [f"requirement_id: {row[0]}, description: {row[1]}" for row in rows]
    )


@tool
def handoff_to_flight_intent_elicitation_tool(
    requirement_id: str,
    task_description: Annotated[str, "Describe what the next agent should do."],
    state: Annotated[MessagesState, InjectedState],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Hand off to flight intent elicitation assistant to, help create/update flight search requirements.
    requirement_id: the id of the flight search requirement to update, obtained from fetch_user_flight_search_requirement. If empty, create a new flight search requirement.
    task_description: describe the task the intent elicitation assistant should do. e.g. "help user create a new flight requirement", "help user update existing flight requirement; update departure airport to be SFO"
    """

    # set a timeout for testing:
    # time.sleep(30)
    # print(
    #     "-------------------------------- timeout for testing --------------------------------"
    # )
    print("-------------------------------- handoff to intent elicitation assistant --------------------------------")
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
        # graph=Command.PARENT,
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