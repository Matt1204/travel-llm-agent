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
from graph_setup import INTENT_GRAPH
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
import sqlite3
from db_connection import db_file

# def create_handoff_to_worker_tool(
#     to_agent_name: str, job_description: str | None = None
# ):
#     name = f"transfer_to_{to_agent_name}"
#     job_description = (
#         job_description or f"Hand off to {to_agent_name} assistant for help."
#     )

#     @tool(name, description=job_description)
#     def handoff_to_worker_tool(
#         task_description: Annotated[str, "Describe what the next agent should do."],
#         state: Annotated[MessagesState, InjectedState],
#     ) -> Command:
#         # Build a minimal, precise input for the next agent
#         agent_input = {
#             **state,  # dict unpacking
#             "messages": [
#                 {"role": "user", "content": "66576657"}
#             ],  # TODO: replace with actual user input
#             "requirement_id": "123321",  # TODO: replace with actual requirement id
#         }
#         return Command(
#             goto=[Send(to_agent_name, agent_input)],  # << key line
#             # graph=Command.PARENT,
#         )

#     return handoff_to_worker_tool

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
    return "\n".join([f"{row[0]}: {row[1]}" for row in rows])
    
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
    agent_input = {
        **state,  # dict unpacking
        "messages": [
            {"role": "user", "content": task_description}
        ],
        "requirement_id": requirement_id,
    }
    return Command(
        goto=[Send(INTENT_GRAPH, agent_input)], 
        update={
            "dialog_state": ["in_intent_elicitation_assistant"],
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Handing off to intent elicitation assistant to help user on task: {task_description}",
                )
            ],
        }
        # graph=Command.PARENT,
    )

# Treated as a tool, to be invoked by the primary assistant
# which will become part of the AIMessage=>tool_name, args when invoked by the primary assistant
class ToFlightAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
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
    example 2: "cancel": True, "reason": "I have fully completed the task.",
    example 3: "cancel": False, "reason": "I need to search the user's emails or calendar for more information.",
    """

    cancel: bool = True
    reason: str

    # class Config:
    #     json_schema_extra = {
    #         "examples": [
    #             {
    #                 "cancel": True,
    #                 "reason": "User changed their mind about the current task.",
    #             },
    #             {
    #                 "cancel": True,
    #                 "reason": "I have fully completed the task.",
    #             },
    #             {
    #                 "cancel": False,
    #                 "reason": "I need to search the user's emails or calendar for more information.",
    #             },
    #         ]
    #     }
