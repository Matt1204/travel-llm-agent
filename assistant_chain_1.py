from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

# from langchain_anthropic import ChatAnthropic
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

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

from langchain_openai import ChatOpenAI


def update_dialog_stack(prev_val: list[str], value: str):
    if value is "pop":
        return prev_val[:-1]
    return prev_val + [value]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        Literal["primary_assistant", "flight_assistant"],
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


# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17", temperature=0.2
)
# llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.2)
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful customer support assistant for Swiss Airlines. 
            Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
            When searching, be persistent. Expand your query bounds if the first search returns no results. 
            If a search comes up empty, expand your search before giving up.
            \n\nCurrent user:\n<User>\n{user_info}\n</User>
            \nCurrent time: {time}.""",
        ),
        # ("placeholder", "{messages}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(time=datetime.now)

tavily_tool = TavilySearch(max_results=1)
safe_tools = [tavily_tool, fetch_user_flight_information, search_flights]
sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
part_1_assistant_chain = primary_assistant_prompt | llm.bind_tools(
    sensitive_tools + safe_tools
)

safe_tools_names = [tool.name for tool in safe_tools]
sensitive_tools_names = [tool.name for tool in sensitive_tools]
