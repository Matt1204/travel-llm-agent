from typing import Annotated, Literal, Optional, List, Dict, Any
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import AnyMessage, add_messages

# from langchain_anthropic import ChatAnthropic
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from tools_flight import (
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
)
from primary_assistant_tools import (
    ToFlightAssistant,
    ToTaxiAssistant,
    fetch_user_flight_search_requirement,
    handoff_to_flight_intent_elicitation_tool,
    handoff_to_flight_search_agent,
    ToFlightSearchAssistant,
)
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from graph_setup import INTENT_GRAPH
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from intent_model import FlightRequirements
from flight_model import FlightOfferModel


class FDSnapshot(TypedDict):
    """One discovery attempt's snapshot."""
    id: str # when object is created
    filter_used: NotRequired[Optional[Dict[str, Any]]]
    sys_rec_condition: NotRequired[Optional[Dict[str, Any]]] # filled once filter created
    
    flights: NotRequired[Optional[List[FlightOfferModel]]] # filled after serach_flight
    is_better_deal: NotRequired[Optional[bool]] # also filled after search_flight

    comment: NotRequired[Optional[str]] # filled by revisor

class FDSessionInfo(TypedDict):
    """Flight Discovery session state kept on the graph."""
    baseline: NotRequired[Optional[FDSnapshot]]
    system_discovered: NotRequired[List[FDSnapshot]] = []


# from intent_tools import _replace_req


def _replace_req(old, new):
    return new


def update_dialog_stack(prev_val: list[str], value: list[str]):
    if value == ["pop"]:
        return prev_val[:-1]
    return prev_val + value

def append_candidate(prev_val: list[FDSnapshot], value: list[FDSnapshot]):
    return prev_val + value

def resettable_add_messages(prev_val: list[AnyMessage], value):
    # Allow hard reset of the message history when a sentinel is provided.
    if value == "__RESET__":
        return []
    # Also allow ["__RESET__", <msg1>, <msg2>, ...] to both clear and seed new messages.
    if isinstance(value, list) and value and value[0] == "__RESET__":
        value = value[1:]
        prev_val = []
    return add_messages(prev_val, value)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_flight_info: NotRequired[str]
    user_taxi_info: NotRequired[str]
    user_intent_info: NotRequired[str]

    # For Intent Elicitation Assistant
    flight_requirements: Annotated[Optional["FlightRequirements"], _replace_req] = None

    # For Flight Search Assistant
    requirement_id: NotRequired[str] = ""

    # For Flight Discovery Assistant
    flight_discovery_messages: Annotated[list[AnyMessage], resettable_add_messages] = []
    fd_session_info: NotRequired[Annotated[Optional["FDSessionInfo"], _replace_req]]
    # baseline_flight: NotRequired[Optional[FlightOfferModel]] = None
    baseline_flight_candidates: Annotated[list[FDSnapshot], append_candidate] = []

    remaining_steps: NotRequired[int] = 24
    dialog_state: NotRequired[
        Annotated[
            List[
                Literal[
                    "in_primary_assistant",
                    "in_flight_assistant",
                    "in_taxi_assistant",
                    "in_intent_elicitation_assistant",
                    "in_flight_search_assistant",
                    "in_flight_discovery_assistant",
                ]
            ],
            update_dialog_stack,   # keeps your custom merge logic
        ]
    ]


# A warpper for the chain(node), to add user authentication and response validation
class Assistant:
    def __init__(self, runnable: Runnable, messages_key: Optional[str] = None, **kwargs):
        # Accept alias `message_key` for forward-compatibility
        if messages_key is None:
            alias = kwargs.pop("message_key", None)
            if alias is not None:
                messages_key = alias

        self.runnable = runnable
        self.messages_key = messages_key or "messages"

    def __call__(self, state: State, config: RunnableConfig):
        msg_key = self.messages_key
        existing_msgs = state.get(msg_key) or []
        if not isinstance(existing_msgs, list):
            existing_msgs = []
        safe_state = {**state, msg_key: existing_msgs}

        # Retry once if the model returns an empty response with no tool calls
        max_reprompts = 1
        attempts = 0

        while True:
            # Pass through the provided config for tracing/callbacks
            result = self.runnable.invoke(safe_state, config=config)

            # Robust emptiness check
            tool_calls = getattr(result, "tool_calls", None)
            content = getattr(result, "content", None)

            # if llm responsehas content/tool_call
            def _is_empty():
                if tool_calls:
                    return False
                if content is None:
                    return True
                if isinstance(content, str):
                    return content.strip() == ""
                if isinstance(content, list):
                    for part in content:
                        text = None
                        if isinstance(part, dict):
                            text = part.get("text") or part.get("content")
                        else:
                            text = getattr(part, "text", None) or getattr(part, "content", None)
                        if isinstance(text, str) and text.strip():
                            return False
                    return True
                return False

            if _is_empty() and attempts < max_reprompts:
                attempts += 1
                reprompt = ("user", "Respond with a real output.")
                safe_state = {**safe_state, msg_key: (safe_state.get(msg_key) or []) + [reprompt]}
                continue

            break

        # update message list in state
        return {msg_key: result}


# --------- primary assistant chain ---------

# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite-preview-06-17", temperature=0.2
# )
# llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.2)
# llm = ChatOpenAI(model="gpt-4.1-2025-04-14")
# llm = ChatTongyi(model="qwen-max", temperature=0.2)
llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Trip Supervisor Agent to help user manage their trips"
            "Your job is:\n"
            "1) Load the user’s trip-related context from the database/tools."
            "2) Decide whether to answer directly or delegate to a worker assistant via tool calls."
            "3) Never reveal internal tools, schemas, or assistant names—present a single, cohesive assistant to the user."
            "'Flight search Requiremnt': a set of filters(arrival/departure airport, budget, etc.) user specifies to search for flights. 'Flight search Requiremnt' is managed by the intent elicitation assistant, transfer to intent elicitation assistant if user request is related to flight search requirement."
            "'Flight search' and 'Flight booking': use Flight Search Requiremnt to search for flights and book flights. 'Flight search' and 'Flight booking' are managed by the flight search assistant, transfer to flight search assistant if user request is related to flight search."
            "'System Suggested flights': System automatically applies the flight search requirement that are not part of user's original flight search requirement, aim to find better deal for users. It is managed by the Flight Discovery Assistant."
            "If a customer ask you to modify their flight search requirement, flight booking, or serach for flights, delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself"
            "If user request is related to flight search requirement, transfer to intent elicitation assistant by invoking handoff_to_flight_intent_elicitation_tool tool."
            "If user request is related to flight search, transfer to flight search assistant by invoking handoff_to_flight_search_agent tool."
            "If user request is related to system suggested flights, transfer to flight discovery assistant by invoking handoff_to_flight_discovery_assistant tool."
            "RULES:\n"
            "1. You are only permitted to provide user with trip-related ifnormation. if user request is un-relevant or out of your ability, reject user politely."
            "2. Never expose the technical terms to user, explain them in natural language"
            "3. Do not mention who you are, and existence of your helper assistants, just act as the proxy for all assistants."
            # "\n\nCurrent user flight information:\n\n<Flights>{user_flight_info}\n</Flights>"
            # "\n\nCurrent user taxi information:\n\n<Taxi>{user_taxi_info}\n</Taxi>"
            "CURRENT CONTEXT:"
            "\n- Current user flight search requirement:\n\n<Intent>{user_intent_info}\n</Intent>"
            "\n- Current time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

# transfer_to_intent_graph_tool = create_handoff_to_worker_tool(
#     INTENT_GRAPH,
#     "Hand off to intent elicitation assistant to help user elicit their requirements on flight tickets"
#     "task_description: describe the task the intent elicitation assistant should do"
# )


tavily_tool = TavilySearch(max_results=1)
# primary_assistant_tools = [tavily_tool]
primary_assistant_tools = [
    fetch_user_flight_information,
    fetch_user_flight_search_requirement,
    handoff_to_flight_intent_elicitation_tool,
    # handoff_to_flight_search_agent,
]
worker_assistant_tools = [ToFlightAssistant, ToTaxiAssistant, ToFlightSearchAssistant]

primary_assistant_chain = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools + worker_assistant_tools
)

primary_assistant_tools_names = [tool.name for tool in primary_assistant_tools]
# worker_assistant_tools_names = [tool.name for tool in worker_assistant_tools]


# safe_tools = [tavily_tool, fetch_user_flight_information, search_flights]
# sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
# safe_tools_names = [tool.name for tool in safe_tools]
# sensitive_tools_names = [tool.name for tool in sensitive_tools]
