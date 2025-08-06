# from graph import builder
from graph import State, RunnableConfig, END, Command, ToolNode, interrupt, StateGraph
from primary_assistant_chain import Assistant
from langchain_core.messages import ToolMessage
from primary_assistant_tools import WorkerCompleteOrEscalate
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Literal
from graph_setup import PRIMARY_ASSISTANT
from flight_assistant_chain import (
    flight_assistant_chain,
    flight_assistant_sensitive_tools,
    flight_assistant_safe_tools,
    flight_assistant_sensitive_tools_names,
    flight_assistant_safe_tools_names,
)
from graph_setup import (
    FLIGHT_ENTRY_NODE,
    FLIGHT_ASSISTANT,
    FLIGHT_ASSISTANT_TOOL_HANDLER,
    FLIGHT_ASSISTANT_RETURN_NODE,
    FLIGHT_ASSISTANT_SENSITIVE_TOOLS,
    FLIGHT_ASSISTANT_SAFE_TOOLS,
)


def flight_entry_node(state: State, config: RunnableConfig):
    print("HITTTTTTT Flight Assistant Entry Node")

    tc_message_template = (
        "The assistant is now the 'flight_assistant'. Reflect on the above conversation between the host assistant and the user.\n"
        "The user's intent is unsatisfied. Use the provided tools to assist the user.\n"
    )
    flight_entry_message = []
    for tc in state["messages"][-1].tool_calls or []:
        args = tc.get("args") or {}
        request = args.get("request")
        if request is not None:
            content = tc_message_template + "Task: " + request
            flight_entry_message.append(
                ToolMessage(tool_call_id=tc["id"], content=content)
            )
        else:
            # maybe skip or log missing request
            continue
    return {"messages": flight_entry_message, "dialog_state": ["in_flight_assistant"]}


def flight_assistant_tool_handler(state: State, config: RunnableConfig) -> Command[
    Literal[
        FLIGHT_ASSISTANT_SENSITIVE_TOOLS,
        FLIGHT_ASSISTANT_SAFE_TOOLS,
        FLIGHT_ASSISTANT_RETURN_NODE,
    ]
]:
    route = tools_condition(state)

    # if the flight assistant no tool call, end graph execution
    if route == END:
        print("!!!!!!!!!! Flight assistant END")
        return Command(goto=END)

    tool_call_list = state["messages"][-1].tool_calls
    tool_name_list = [tc["name"] for tc in tool_call_list]

    # if the tool is WorkerCompleteOrEscalate, return to the primary assistant
    if WorkerCompleteOrEscalate.__name__ in tool_name_list:
        print("!!!!!!!!!! Flight assistant escalate back to primary")
        return Command(goto=FLIGHT_ASSISTANT_RETURN_NODE)

    tool_name = tool_name_list[0]
    # if the tool is sensitive, ask the user for approval
    if tool_name in flight_assistant_sensitive_tools_names:
        print("!!!!! HIIIT flight assistant sensitive tool")
        approval = interrupt(
            {
                "text_to_revise": "Do you want to proceed with the tool call: "
                + tool_name
                + "?"
            }
        )
        if approval == True:
            print("!!!!! HIIIT flight sensitive tool approved")
            return Command(goto=FLIGHT_ASSISTANT_SENSITIVE_TOOLS)
        else:
            print("!!!!! flight assistant sensitive tool DENIED, back to FLIGHT_ASSISTANT")
            return Command(
                goto=FLIGHT_ASSISTANT,
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                            content=f"Tool call denied by user. reason: {approval}. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
            )
    elif tool_name in flight_assistant_safe_tools_names:
        print("!!!!! HIIIT flight assistant safe tool")
        return Command(goto=FLIGHT_ASSISTANT_SAFE_TOOLS)
    else:
        print("!!!!! flight assistant Unknown tool")
        return Command(
            goto=FLIGHT_ASSISTANT,
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                        content=f"Tool call failed, tool name not found. Continue assisting, accounting for the user's input.",
                    )
                ]
            },
        )


def flight_assistant_return_node(state: State, config: RunnableConfig):
    print("HITTTTTTT Flight Assistant Return Node")
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "messages": messages,
        "dialog_state": ["pop"],
    }




def register_flight_graph(builder: StateGraph):
    builder.add_node(FLIGHT_ENTRY_NODE, flight_entry_node)
    builder.add_node(FLIGHT_ASSISTANT, Assistant(flight_assistant_chain))
    builder.add_edge(FLIGHT_ENTRY_NODE, FLIGHT_ASSISTANT)

    # "flight assistant node" alwats steps into "flight_assistant_tool_handler"
    builder.add_node(FLIGHT_ASSISTANT_TOOL_HANDLER, flight_assistant_tool_handler)
    builder.add_edge(FLIGHT_ASSISTANT, FLIGHT_ASSISTANT_TOOL_HANDLER)

    builder.add_node(
        FLIGHT_ASSISTANT_SENSITIVE_TOOLS, ToolNode(flight_assistant_sensitive_tools)
    )
    builder.add_node(FLIGHT_ASSISTANT_SAFE_TOOLS, ToolNode(flight_assistant_safe_tools))
    builder.add_edge(FLIGHT_ASSISTANT_SAFE_TOOLS, FLIGHT_ASSISTANT)
    builder.add_edge(FLIGHT_ASSISTANT_SENSITIVE_TOOLS, FLIGHT_ASSISTANT)

    builder.add_node(FLIGHT_ASSISTANT_RETURN_NODE, flight_assistant_return_node)
    builder.add_edge(FLIGHT_ASSISTANT_RETURN_NODE, PRIMARY_ASSISTANT)