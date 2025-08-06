from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatTongyi
from datetime import datetime
from tools_flight import (
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
)

from primary_assistant_tools import WorkerCompleteOrEscalate
from langchain_openai import ChatOpenAI

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite-preview-06-17", temperature=0.2
# )
# llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1)
llm = ChatTongyi(model="qwen-max", temperature=0.1)

flight_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling flight updates."
            "The primary assistant delegates work to you whenever the user needs help updating their bookings."
            "Confirm the updated flight details with the customer and inform them of any additional fees."
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "Ask user for permission before updating the flight. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\n\nCurrent user flight information:\n\n{user_info}\n</user_info>"
            "\nCurrent time: {time}."
            "\n\nRules you must follow:\n"
            "1. If the user needs help, and none of your tools are appropriate for it, then call 'WorkerCompleteOrEscalate' to pass dialog to the host assistant. Do not waste the user's time. Do not make up invalid tools or functions or parameters.\n"
            "2. If you think you have completed the current task you are assigned with, you MUST ONLY call the 'WorkerCompleteOrEscalate' tool to complete the dialog\n"
            "3. If you you do not have enough data to complete the current task, ask the user for the missing information.\n"
            "4. if you want to call a tool, return exactly one tool call per response; if multiple tools are required, choose the most urgent and wait for the observation before deciding the next tool call.\n"
            "5. Do not mention who you are, just act as the proxy for the assistant.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

flight_assistant_safe_tools = [search_flights, fetch_user_flight_information]
flight_assistant_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
flight_assistant_tools = flight_assistant_safe_tools + flight_assistant_sensitive_tools

flight_assistant_chain = flight_assistant_prompt | llm.bind_tools(
    flight_assistant_tools + [WorkerCompleteOrEscalate]
)

flight_assistant_safe_tools_names = [tool.name for tool in flight_assistant_safe_tools]
flight_assistant_sensitive_tools_names = [
    tool.name for tool in flight_assistant_sensitive_tools
]
