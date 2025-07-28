from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from assistant_chain_1 import (
    Assistant,
    State,
    part_1_assistant_chain,
    sensitive_tools,
    safe_tools,
    sensitive_tools_names,
    safe_tools_names,
)
from langchain_core.runnables import RunnableConfig
from tools_flight import fetch_user_flight_information
import json
import uuid
from langgraph.types import interrupt, Command

# from langgraph.graph import GraphRunInterrupted
from langchain_core.messages import ToolMessage

builder = StateGraph(State)


def fetch_user_info(state: State, config: RunnableConfig):
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured, cannot fetch user info.")
    user_info = fetch_user_flight_information.invoke({})

    return {"user_info": json.dumps(user_info)}


def tool_handler(state: State, config: RunnableConfig):
    ai_message = state["messages"][-1]

    first_tool_call = ai_message.tool_calls[0]
    tool_call_id = ai_message.tool_calls[0]["id"]
    tool_name = first_tool_call["name"]

    if tool_name in sensitive_tools_names:
        approval = interrupt(
            {
                "text_to_revise": "Do you want to proceed with the tool call: "
                + tool_name
                + "?"
            }
        )
        if approval == True:
            return Command(goto="sensitive_tools")
        else:
            return Command(
                goto="assistant",
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=state["messages"][-1].tool_call[0]["id"],
                            content=f"Tool call denied by user. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
            )
    elif tool_name in safe_tools_names:
        return Command(goto="safe_tools")
    else:
        return Command(
            goto="assistant",
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id=state["messages"][-1].tool_call[0]["id"],
                        content=f"Tool call failed, tool name not found. Continue assisting, accounting for the user's input.",
                    )
                ]
            },
        )


builder.add_node("fetch_user_info", fetch_user_info)
builder.add_node("assistant", Assistant(part_1_assistant_chain))
builder.add_node("tool_handler", tool_handler)
builder.add_node("sensitive_tools", ToolNode(sensitive_tools))
builder.add_node("safe_tools", ToolNode(safe_tools))

builder.add_edge(START, "fetch_user_info")
builder.add_edge("fetch_user_info", "assistant")
builder.add_edge("sensitive_tools", "assistant")
builder.add_edge("safe_tools", "assistant")

def continue_after_thought(state: State, config: RunnableConfig):
    if (
        hasattr(state["messages"][-1], "tool_calls")
        and len(state["messages"][-1].tool_calls) > 0
    ):
        print("Conditional edge: Tool call, GO to tool_handler")
        return "tool_handler"
    else:
        print("Conditional edge: No tool call, END ......")
        return END

builder.add_conditional_edges(
    "assistant",
    continue_after_thought,
    {"tool_handler": "tool_handler", END: END},
)


# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
# graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
