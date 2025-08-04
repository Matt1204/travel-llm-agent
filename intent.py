from __future__ import annotations

from typing import List, Optional, Literal
from typing_extensions import Annotated, TypedDict, NotRequired
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langchain_core.messages import ToolMessage

from langchain_core.runnables import RunnableConfig

from graph import StateGraph, END, START, Command

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState

from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from langchain_community.chat_models import ChatTongyi
from intent_model import FlightRequirements

# ---------------------------------------------------------------------------
# 2. State definition
# ---------------------------------------------------------------------------
# class FlightRequirements(BaseModel):
#     """User's flight requirements for the flight search. All fields optional until elicited."""

#     departure_airport: Optional[str] = Field(
#         default=None, description="IATA/ICAO code of departure airport, e.g. 'YUL'."
#     )
#     arrival_airport: Optional[str] = Field(
#         default=None, description="IATA/ICAO code of arrival airport, e.g. 'YYZ'."
#     )
#     departure_time_window: Optional[list[datetime]] = Field(
#         default=None,
#         description="[start_datetime, end_datetime] for departure. e.g. [datetime(2025, 8, 1, 10, 0), datetime(2025, 8, 2, 20, 0)]",
#     )
#     num_passengers: Optional[int] = Field(
#         default=None, description="Number of passengers. e.g. 2"
#     )
#     budget_window: Optional[list[int]] = Field(
#         default=None, description="[min_budget, max_budget] in CAD. e.g. [400, 600]"
#     )

# class FlightReqWithLevels(BaseModel):

#     budget_window: Optional[list[int]] = Field(
#         default=None, description="[min_budget, max_budget] in CAD. e.g. [400, 600]"
#     )


class IntentElicitationState(AgentState):
    """Graph state = standard AgentState + custom FlightRequirements."""

    flight_requirements: NotRequired["FlightRequirements"]


# ---------------------------------------------------------------------------
# 3. Hooks
# ---------------------------------------------------------------------------
def pre_model_hook(state: IntentElicitationState) -> dict:
    """Trim messages just for the model input without mutating stored history."""
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    # Return under `llm_input_messages` to avoid overwriting the graph state's history
    return {"llm_input_messages": trimmed_messages}


def post_model_hook(state: IntentElicitationState) -> IntentElicitationState:
    """Very naive guardrail - could add PII redaction, audit logging, etc."""
    return state


# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------
@tool
def add_departure_airport_priority(
    priority: int,
    value: list[str],
    *,
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add an departure airport with priority level to the FlightRequirements.
    priority: the priority level to add. e.g. 1(highest), 2, 3, 4
    value: the IATA/ICAO code of departure airport, to add. e.g. ['YUL', 'YHU']
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.add_requirement_priority("departure_airport", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                    content=f"Departure airport priority {priority} added: {value}",
                )
            ],
        }
    )


@tool
def add_arrival_airport_priority(
    priority: int,
    value: list[str],
    *,
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add an arrival airport with priority level to the FlightRequirements.
    priority: the priority level to add. e.g. 1(highest), 2, 3, 4
    value: the IATA/ICAO code of arrival airport, to add. e.g. ['YYZ', 'YTZ']
    """
    current_req = state.get("flight_requirements", FlightRequirements())
    current_req.add_requirement_priority("arrival_airport", priority, value)
    return Command(
        update={
            "flight_requirements": current_req,
            "messages": [
                ToolMessage(
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
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
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add a departure time window with priority level to the FlightRequirements.
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
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
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
    state: Annotated[IntentElicitationState, InjectedState],
    config: RunnableConfig,
) -> Command:
    """Add a budget boundary with priority level to the FlightRequirements.
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
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                    content=f"Budget priority {priority} added: {value}",
                )
            ],
        }
    )


# @tool
# def update_flight_requirements(
#     departure_airport: Optional[str] = None,
#     arrival_airport: Optional[str] = None,
#     departure_time_window: Optional[list[datetime]] = None,
#     num_passengers: Optional[int] = None,
#     budget_window: Optional[list[int]] = None,
#     *,
#     state: Annotated[IntentElicitationState, InjectedState],
#     config: RunnableConfig,
# ) -> Command:
#     """
#     Update the flight requirements in graph state. Provide any subset of fields.
#     departure_airport: IATA/ICAO code of departure airport, e.g. 'YUL'.
#     arrival_airport: IATA/ICAO code of arrival airport, e.g. 'YYZ'.
#     departure_time_window: [start_datetime, end_datetime] for departure. e.g. [datetime(2025, 8, 1, 10, 0), datetime(2025, 8, 2, 20, 0)]
#     num_passengers: Number of passengers. e.g. 2
#     budget_window(optional): [min_budget, max_budget] in CAD. e.g. [400, 600]
#     Return: a Command that updates `flight_requirements` and appends a tool message.
#     """
#     current = state.get("flight_requirements", FlightRequirements())
#     updated = FlightRequirements(
#         departure_airport=departure_airport or current.departure_airport,
#         arrival_airport=arrival_airport or current.arrival_airport,
#         departure_time_window=departure_time_window or current.departure_time_window,
#         num_passengers=(
#             num_passengers if num_passengers is not None else current.num_passengers
#         ),
#         budget_window=budget_window or current.budget_window,
#     )
#     summary = "Flight requirements updated successfully. updated fields: "
#     for field, value in updated.model_dump().items():
#         if value is not None:
#             summary += f"{field}={value}, "
#     return Command(
#         update={
#             "flight_requirements": updated,
#             "messages": [
#                 ToolMessage(
#                     tool_call_id=state["messages"][-1].tool_calls[0]["id"],
#                     content=summary,
#                 )
#             ],
#         }
#     )


# ---------------------------------------------------------------------------
# 4. Prompt
# ---------------------------------------------------------------------------
intent_elicitation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful flight assistant. You are tasked to elicit a user's flight requirements and fill a FlightRequirements object, you should not offer to help user with any other things\n"
            "Each flight criteria in the FlightRequirements has requirement values labeled with priority levels(priority_1, priority_2), this priority level is used to determine a requirements value's importance/urgency/preference to user.\n"
            "Requirement with larger number(e.g. priority_1) is user's most desired/preferred/important. requirement with smaller number(e.g. priority_4) is user's less desired/preferred/important .\n"
            "1) First, guide user to fill in all 'priority_1' values. Ask the user to describe their most desired/preferred/urgent needs; then update flight requirements via the 'create_xxx_priority' tools.\n"
            "2) Ask follow-ups to fill missing fields. Do not ask for values already filled. guide user to fill in all 'priority_1' values before moving to lower priority values(e.g. priority_2, priority_3).\n"
            "3) Ask user if they accept other requirement values(less desired values, e.g. priority_2, priority_3), use an example to explain: 'If you can accpet leaving 1 day after PRIORITY_1_DEPARTURE_TIME, I may find you more options'. If they do, ask them to describe their needs; then add flight requirements via the 'create_xxx_priority' tools.\n"
            "4) When complete, read back the requirements and ask user to explicitly confirm by using keyword 'confirm'.\n"
            "Current FlightRequirements:\n{flight_requirements}\n"
            "Current time: {time}\n\n"
            "Rules:\n"
            "- You are only allowed to assist user to elicit their flight requirement specified in FlightRequirements. You should not offer to help with any other things.\n"
            "- You must update the requirements with the tool once user provide any new information. Do not make up values. do not waste the user's time.\n"
            "- If you are not 100% sure about the user's input, ask them to confirm by using keyword 'confirm'. e.g if user only specifies the city of departure, you will need to ask user to confirm the airport and IATA/ICAO code.\n"
            "- Only ask for fields defined in FlightRequirements.\n"
            "- Do not explain 'priority' using 'priority_1' or 'priority_2'. Use keywords like 'most desired' or 'preferred' or 'urgent' instead.\n"
            # "- Call at most ONE tool per response; if more are needed, wait for the observation.\n"
            "- You MUST NOT answer any user input that is irrelevant to the elicitation of flight ticket requirements. e.g. if the user says 'how is the weather today?', you MUST politely refuse to answer it and direct the user to flight ticket requirements.\n"
            "- Do not mention who you are; act as the assistant.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now().isoformat())


# ---------------------------------------------------------------------------
# 5. Agent definition
# ---------------------------------------------------------------------------
class ResponseModel(BaseModel):
    """Response model for the intent elicitation agent."""

    response: str = Field(description="The response to the user's input")
    flight_requirements: FlightRequirements = Field(
        description="The flight requirements with updated values"
    )


llm = ChatTongyi(model="qwen-plus", temperature=0.1)
# llm = llm.with_structured_output(ResponseModel)
checkpointer = InMemorySaver()


def fetch_requirements_node(state: IntentElicitationState, config: RunnableConfig):
    """Fetch the flight requirements from the database."""
    # use dummy data for now

    current_requirements = state.get("flight_requirements", None)
    if current_requirements is not None:
        return {"flight_requirements": current_requirements}

    requirements = FlightRequirements()

    return {
        "flight_requirements": requirements,
    }

tools = [
    add_departure_airport_priority,
    add_arrival_airport_priority,
    add_departure_time_priority,
    add_budget_priority,
]

intent_elicitation_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=intent_elicitation_prompt,
    state_schema=IntentElicitationState,
    # pre_model_hook=pre_model_hook,
    # post_model_hook=post_model_hook,
    # debug=True,
)

# ---------------------------------------------------------------------------
# 6. Compose overall graph – Router → Worker(s) → END
# ---------------------------------------------------------------------------
graph = StateGraph(IntentElicitationState)

# Add nodes
graph.add_node("fetch_requirements_node", fetch_requirements_node)
graph.add_edge(START, "fetch_requirements_node")

graph.add_node("intent_elicitation_agent", intent_elicitation_agent)
graph.add_edge("fetch_requirements_node", "intent_elicitation_agent")

graph.add_edge("intent_elicitation_agent", END)

# Compile = runnable LangGraph
graph = graph.compile(checkpointer=checkpointer)


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

        stream = graph.stream(
            {"messages": [("user", q)]}, config, stream_mode=["values"]
        )
        for event in stream:
            event[-1]["messages"][
                -1
            ].pretty_print()  # chunk: contains accumulated messages
        cur_state = graph.get_state(config)
        # print(cur_state["messages"][-1].pretty_print())
        if "flight_requirements" in cur_state.values:
            print(
                "\n[DEBUG] Current requirements:",
                cur_state.values["flight_requirements"].to_json(),
            )
