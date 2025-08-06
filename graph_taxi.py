from graph import State, RunnableConfig, END, Command, ToolNode, interrupt, StateGraph
from primary_assistant_chain import Assistant
from typing import Literal
from langchain_core.messages import ToolMessage
from taxi_assistant_chain import taxi_assistant_chain
from langgraph.prebuilt import tools_condition, ToolNode
from primary_assistant_tools import WorkerCompleteOrEscalate
from graph_setup import (
    TAXI_ASSISTANT_SENSITIVE_TOOLS,
    TAXI_ASSISTANT_SAFE_TOOLS,
    TAXI_ASSISTANT_TOOL_HANDLER,
    TAXI_ASSISTANT_RETURN_NODE,
    TAXI_ENTRY_NODE,
    TAXI_ASSISTANT,
    PRIMARY_ASSISTANT,
)
from taxi_assistant_chain import (
    taxi_assistant_safe_tools,
    taxi_assistant_sensitive_tools,
    taxi_assistant_safe_tools_names,
    taxi_assistant_sensitive_tools_names,
)


def taxi_entry_node(state: State, config: RunnableConfig):
    print("HITTTTTTT Taxi Assistant Entry Node")

    tc_message_template = (
        "The assistant is now the 'taxi_assistant'. Reflect on the above conversation between the host assistant and the user.\n"
        "The user's intent is unsatisfied. Use the provided tools to assist the user.\n"
    )
    taxi_entry_message = []
    for tc in state["messages"][-1].tool_calls or []:
        args = tc.get("args") or {}
        request = args.get("request")
        if request is not None:
            content = tc_message_template + "Task: " + request
            taxi_entry_message.append(
                ToolMessage(tool_call_id=tc["id"], content=content)
            )
        else:
            # maybe skip or log missing request
            continue
    return {"messages": taxi_entry_message, "dialog_state": ["in_taxi_assistant"]}


def taxi_assistant_tool_handler(state: State, config: RunnableConfig) -> Command[
    Literal[
        TAXI_ASSISTANT_SENSITIVE_TOOLS,
        TAXI_ASSISTANT_SAFE_TOOLS,
        TAXI_ASSISTANT_RETURN_NODE,
    ]
]:
    route = tools_condition(state)

    # if the taxi assistant no tool call, end graph execution
    if route == END:
        print("!!!!!!!!!! Taxi assistant END")
        return Command(goto=END)

    tool_call_list = state["messages"][-1].tool_calls
    tool_name_list = [tc["name"] for tc in tool_call_list]

    # if the tool is WorkerCompleteOrEscalate, return to the primary assistant
    if WorkerCompleteOrEscalate.__name__ in tool_name_list:
        print("!!!!!!!!!! Taxi assistant escalate back to primary")
        return Command(goto=TAXI_ASSISTANT_RETURN_NODE)

    tool_name = tool_name_list[0]
    # if the tool is sensitive, ask the user for approval
    if tool_name in taxi_assistant_sensitive_tools_names:
        print("!!!!! HIIIT taxi assistant sensitive tool")
        user_input = interrupt(
            {
                "text_to_revise": "Do you want to proceed with the tool call: "
                + tool_name
                + "?"
            }
        )
        if user_input == True:
            print("!!!!! HIIIT sensitive tool approved")
            return Command(goto=TAXI_ASSISTANT_SENSITIVE_TOOLS)
        else:
            print("!!!!! sensitive tool DENIED, to TAXI_ASSISTANT")
            return Command(
                goto=TAXI_ASSISTANT,
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                            content=f"Tool call denied by user. reason: {user_input}. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
            )
    elif tool_name in taxi_assistant_safe_tools_names:
        print("!!!!! HIIIT taxi assistant safe tool")
        return Command(goto=TAXI_ASSISTANT_SAFE_TOOLS)
    else:
        print("!!!!! flight assistant Unknown tool")
        return Command(
            goto=TAXI_ASSISTANT,
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                        content=f"Tool call failed, tool name not found. Continue assisting, accounting for the user's input.",
                    )
                ]
            },
        )


def taxi_assistant_return_node(state: State, config: RunnableConfig):
    print("HITTTTTTT Taxi Assistant Return Node")
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


def register_taxi_graph(builder: StateGraph):
    builder.add_node(TAXI_ENTRY_NODE, taxi_entry_node)
    builder.add_node(TAXI_ASSISTANT, Assistant(taxi_assistant_chain))
    builder.add_edge(TAXI_ENTRY_NODE, TAXI_ASSISTANT)

    builder.add_node(TAXI_ASSISTANT_TOOL_HANDLER, taxi_assistant_tool_handler)
    builder.add_edge(TAXI_ASSISTANT, TAXI_ASSISTANT_TOOL_HANDLER)

    builder.add_node(
        TAXI_ASSISTANT_SENSITIVE_TOOLS, ToolNode(taxi_assistant_sensitive_tools)
    )
    builder.add_node(TAXI_ASSISTANT_SAFE_TOOLS, ToolNode(taxi_assistant_safe_tools))
    builder.add_edge(TAXI_ASSISTANT_SAFE_TOOLS, TAXI_ASSISTANT)
    builder.add_edge(TAXI_ASSISTANT_SENSITIVE_TOOLS, TAXI_ASSISTANT)

    builder.add_node(TAXI_ASSISTANT_RETURN_NODE, taxi_assistant_return_node)
    builder.add_edge(TAXI_ASSISTANT_RETURN_NODE, PRIMARY_ASSISTANT)
