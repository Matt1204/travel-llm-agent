from typing import Annotated, Literal, Optional, List
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
)
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from graph_setup import INTENT_GRAPH
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from intent_model import FlightRequirements

# from intent_tools import _replace_req


def _replace_req(old, new):
    return new


def update_dialog_stack(prev_val: list[str], value: list[str]):
    if value == ["pop"]:
        return prev_val[:-1]
    return prev_val + value


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_flight_info: NotRequired[str]
    user_taxi_info: NotRequired[str]
    user_intent_info: NotRequired[str]

    flight_requirements: Annotated[Optional["FlightRequirements"], _replace_req] = None
    requirement_id: NotRequired[str] = ""
    remaining_steps: NotRequired[int] = 24

    dialog_state: NotRequired[
        Annotated[
            List[
                Literal[
                    "in_primary_assistant",
                    "in_flight_assistant",
                    "in_taxi_assistant",
                    "in_intent_elicitation_assistant",
                ]
            ],
            update_dialog_stack,   # keeps your custom merge logic
        ]
    ]


# A warpper for the chain(node), to add user authentication and response validation
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # feeding State to chain, State has "messages" and "user_info", which will be appended to system prompt

            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# --------- primary assistant chain ---------

# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite-preview-06-17", temperature=0.2
# )
# llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.2)
llm = ChatOpenAI(model="gpt-4.1-2025-04-14")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant to help user maage their trips. "
            "Your primary role is to search for flight search requirements/flight information to answer customer queries."
            "If a customer requests to create/update/cancel a flight, flight search requirement, or taxi, delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            "Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "If the user requests to search for their flights, you can use the fetch_user_flight_search_requirement tool to fetch the user's flight search requirements from the database. "
            "If user requests to search for their flight search requirements(which is a set of criteria user specifies to search for flights), you can use the fetch_user_flight_search_requirement tool"
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            # " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n\n<Flights>{user_flight_info}\n</Flights>"
            # "\n\nCurrent user taxi information:\n\n<Taxi>{user_taxi_info}\n</Taxi>"
            "\n\nCurrent user flight search(requirement) information:\n\n<Intent>{user_intent_info}\n</Intent>"
            "\nCurrent time: {time}.",
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
]
worker_assistant_tools = [ToFlightAssistant, ToTaxiAssistant]

primary_assistant_chain = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools + worker_assistant_tools
)

primary_assistant_tools_names = [tool.name for tool in primary_assistant_tools]
# worker_assistant_tools_names = [tool.name for tool in worker_assistant_tools]


# safe_tools = [tavily_tool, fetch_user_flight_information, search_flights]
# sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
# safe_tools_names = [tool.name for tool in safe_tools]
# sensitive_tools_names = [tool.name for tool in sensitive_tools]
