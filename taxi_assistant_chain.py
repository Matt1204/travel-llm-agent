from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from tools_taxi import (
    fetch_user_taxi_requests,
    create_taxi_request,
    remove_taxi_request,
)

from pydantic_tools import WorkerCompleteOrEscalate
from langchain_openai import ChatOpenAI

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite-preview-06-17", temperature=0.2
# )
llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.1)

taxi_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling taxi orders. "
            "The primary assistant delegates work to you whenever the user needs help with their taxi requests. "
            "Confirm the all updated taxi request details with the customer and inform them of any additional fees. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\n\nCurrent user taxi request information:\n\n{user_info}\n</Flights>"
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' call "WorkerCompleteOrEscalate" to pass dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions or parameters.\n'
            "If you think you have completed the current task, MUST call the WorkerCompleteOrEscalate tool to complete the dialog.\n"
            "If you you do not have enough data to complete the task, ask the user for the missing information.\n"
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

taxi_assistant_safe_tools = [fetch_user_taxi_requests]
taxi_assistant_sensitive_tools = [remove_taxi_request, create_taxi_request]
taxi_assistant_tools = taxi_assistant_safe_tools + taxi_assistant_sensitive_tools

taxi_assistant_chain = taxi_assistant_prompt | llm.bind_tools(
    taxi_assistant_tools + [WorkerCompleteOrEscalate]
)

taxi_assistant_safe_tools_names = [tool.name for tool in taxi_assistant_safe_tools]
taxi_assistant_sensitive_tools_names = [
    tool.name for tool in taxi_assistant_sensitive_tools
]
