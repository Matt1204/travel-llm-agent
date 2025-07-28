# from graph import builder
from graph import State, RunnableConfig, END, Command, ToolNode, interrupt, StateGraph
from primary_assistant_chain import Assistant
from langchain_core.messages import ToolMessage
from pydantic_tools import WorkerCompleteOrEscalate
from langgraph.prebuilt import tools_condition, ToolNode

from pydantic_tools import WorkerCompleteOrEscalate
from graph_setup import PRIMARY_ASSISTANT
from flight_assistant_chain import (
    flight_assistant_chain,
    flight_assistant_sensitive_tools,
    flight_assistant_safe_tools,
    flight_assistant_sensitive_tools_names,
    flight_assistant_safe_tools_names,
)


def flight_entry_node(state: State, config: RunnableConfig):
    print("HITTTTTTT Flight Assistant Entry Node")
    flight_entry_message = ToolMessage(
        tool_call_id=state["messages"][-1].tool_calls[0]["id"],
        content=f"""The assistant is now the {"flight_assistant"}. Reflect on the above conversation between the host assistant and the user.
        The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {"flight_assistant"},
        and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool.
        If the user changes their mind or needs help for other tasks, call the WorkerCompleteOrEscalate tool to let the primary host assistant take control.
        Do not mention who you are - just act as the proxy for the assistant.""",
    )
    return {"messages": [flight_entry_message], "dialog_state": ["in_flight_assistant"]}



def flight_assistant_tool_handler(state: State, config: RunnableConfig):
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
        return Command(goto="flight_assistant_return_node")

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
            print("!!!!! HIIIT sensitive tool approved")
            return Command(goto="flight_assistant_sensitive_tools")
        else:
            print("!!!!! sensitive tool DENIED")
            return Command(
                goto="flight_entry_node",
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                            content=f"Tool call denied by user. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
            )
    elif tool_name in flight_assistant_safe_tools_names:
        print("!!!!! HIIIT flight assistant safe tool")
        return Command(goto="flight_assistant_safe_tools")
    else:
        print("!!!!! flight assistant Unknown tool")
        return Command(
            goto="flight_assistant_return_node",
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
        "dialog_statexw": ["pop"],
    }




def register_flight_graph(builder: StateGraph):
    builder.add_node("flight_entry_node", flight_entry_node)

    builder.add_edge("flight_entry_node", "flight_assistant")
    builder.add_node("flight_assistant", Assistant(flight_assistant_chain))

    # "flight assistant node" alwats steps into "flight_assistant_tool_handler"
    builder.add_node("flight_assistant_tool_handler", flight_assistant_tool_handler)
    builder.add_edge("flight_assistant", "flight_assistant_tool_handler")

    builder.add_node(
        "flight_assistant_sensitive_tools", ToolNode(flight_assistant_sensitive_tools)
    )
    builder.add_node("flight_assistant_safe_tools", ToolNode(flight_assistant_safe_tools))
    builder.add_edge("flight_assistant_safe_tools", "flight_assistant")
    builder.add_edge("flight_assistant_sensitive_tools", "flight_assistant")

    builder.add_node("flight_assistant_return_node", flight_assistant_return_node)
    builder.add_edge("flight_assistant_return_node", PRIMARY_ASSISTANT)