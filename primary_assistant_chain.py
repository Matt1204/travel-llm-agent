from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
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
from pydantic_tools import ToFlightAssistant, ToTaxiAssistant
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi

def update_dialog_stack(prev_val: list[str], value: list[str]):
    if value == ["pop"]:
        return prev_val[:-1]
    return prev_val + value


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        Literal["in_primary_assistant", "in_flight_assistant", "in_taxi_assistant"],
        update_dialog_stack,
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
llm = ChatTongyi(model="qwen-plus", temperature=0.1)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for airlines. "
            "Your primary role is to search for flight information to answer customer queries. "
            "If a customer requests to update or cancel a flight and taxi, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user booking information:\n\n{user_info}\n</Flights>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


tavily_tool = TavilySearch(max_results=1)
# primary_assistant_tools = [tavily_tool]
primary_assistant_tools = [fetch_user_flight_information]
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
