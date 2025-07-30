from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from tools_taxi import (
    fetch_user_taxi_requests,
    create_taxi_request,
    remove_taxi_request,
    update_taxi_request,
)
from langchain_community.llms import Tongyi

from pydantic_tools import WorkerCompleteOrEscalate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite-preview-06-17", temperature=0.2
# )
# llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.1)
llm = ChatTongyi(model="qwen-plus", temperature=0.1)

taxi_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling taxi orders. Remember: you are 'taxi_assistant' "
            "The primary assistant delegates work to you whenever the user needs help with their taxi requests. "
            "Confirm the all updated taxi request details with the customer and inform them of any additional fees. "
            "The taxi request is not complete until you have successfully invoked the appropriate tool."
            "If the user changes their mind or needs help for other tasks, call the WorkerCompleteOrEscalate tool to let the primary host assistant take control.\n"
            "\n\nCurrent user taxi request information:\n\n{user_info}\n</user_info>"
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

taxi_assistant_safe_tools = [fetch_user_taxi_requests]
taxi_assistant_sensitive_tools = [
    remove_taxi_request,
    create_taxi_request,
    update_taxi_request,
]
taxi_assistant_tools = taxi_assistant_safe_tools + taxi_assistant_sensitive_tools

taxi_assistant_chain = taxi_assistant_prompt | llm.bind_tools(
    taxi_assistant_tools + [WorkerCompleteOrEscalate]
)

taxi_assistant_safe_tools_names = [tool.name for tool in taxi_assistant_safe_tools]
taxi_assistant_sensitive_tools_names = [
    tool.name for tool in taxi_assistant_sensitive_tools
]
