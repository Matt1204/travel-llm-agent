from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from primary_assistant_chain import (
    Assistant,
    State,
    primary_assistant_chain,
    primary_assistant_tools,
    primary_assistant_tools_names,
)
from langchain_core.runnables import RunnableConfig
from tools_flight import fetch_user_flight_information
import json
import uuid
from langgraph.types import interrupt, Command

# from langgraph.graph import GraphRunInterrupted
from langchain_core.messages import ToolMessage, AIMessage

from primary_assistant_tools import (
    ToFlightAssistant,
    ToTaxiAssistant,

    HandoffToFlightDiscoveryAgent,
    HandoffToFlightSearchAgent
)
from flight_discovery_agent import register_fd_baseline_graph
from fd_discovery_graph import register_fd_discovery_graph

# from intent import intent_graph
# from graph_flight import register_flight_graph
# from graph_taxi import register_taxi_graph
from intent import register_intent_graph
from tools_taxi import fetch_user_taxi_requests
from graph_setup import (
    FETCH_USER_INFO_NODE,
    PRIMARY_ASSISTANT,
    PRIMARY_ASSISTANT_TOOLS_NODE,
    TAXI_ENTRY_NODE,
    TAXI_ASSISTANT,
    INTENT_ENTRY_NODE,
    FLIGHT_SEARCH_INVOKE_NODE,
    FLIGHT_SEARCH_HANDOFF_NODE,
    FD_BASELINE_ENTRY_NODE,
)
from primary_assistant_tools import fetch_user_flight_search_requirement
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState

from flight_search_agent import flight_search_graph


builder = StateGraph(State)


def fetch_user_info_node(state: State, config: RunnableConfig):
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured, cannot fetch user info.")
    # user_flight_info = fetch_user_flight_information.invoke({})
    # user_taxi_info = fetch_user_taxi_requests.invoke({})
    search_requirement_info = fetch_user_flight_search_requirement.invoke({})
    # user_info = {
    #     "flight_info": user_flight_info,
    #     "taxi_info": user_taxi_info,
    #     "intent_info": user_intent_info,
    # }
    # formatted_user_info = json.dumps(user_info, indent=2)
    # return {"user_info": formatted_user_info}  # user_info is a string
    return {
        # "user_flight_info": json.dumps(user_flight_info, indent=2),
        # "user_taxi_info": json.dumps(user_taxi_info, indent=2),
        "search_requirement_info": json.dumps(search_requirement_info, indent=2),
    }


# ------ "sub-graph" for primary assistant ------
builder.add_node(FETCH_USER_INFO_NODE, fetch_user_info_node)
builder.add_edge(START, FETCH_USER_INFO_NODE)


def decision_after_fetch_user_info(state: State, config: RunnableConfig):
    dialog_state_list = state.get("dialog_state")
    if not dialog_state_list:
        # print("No dialog state, to Primary Assistant")
        return PRIMARY_ASSISTANT
    dialog_state = dialog_state_list[-1]
    if dialog_state == "in_intent_elicitation_assistant":
        return INTENT_ENTRY_NODE
    elif dialog_state == "in_flight_search_assistant":
        return FLIGHT_SEARCH_INVOKE_NODE
    elif dialog_state == "in_flight_discovery_assistant":
        return FD_BASELINE_ENTRY_NODE
    else:
        return PRIMARY_ASSISTANT


builder.add_conditional_edges(
    FETCH_USER_INFO_NODE,
    decision_after_fetch_user_info,
    {
        PRIMARY_ASSISTANT: PRIMARY_ASSISTANT,
        # TAXI_ASSISTANT: TAXI_ASSISTANT,
        INTENT_ENTRY_NODE: INTENT_ENTRY_NODE,
        # "flight_assistant": "flight_assistant",
        # FD_BASELINE_ENTRY_NODE: FD_BASELINE_ENTRY_NODE,
        FLIGHT_SEARCH_INVOKE_NODE: FLIGHT_SEARCH_INVOKE_NODE,
    },
)

# builder.add_edge("fetch_user_info", "assistant")
builder.add_node(PRIMARY_ASSISTANT, Assistant(primary_assistant_chain))


def decision_after_primary_thought(state: State, config: RunnableConfig):
    # if the primary assistant has a tool call, delegate to the appropriate tool
    if (
        hasattr(state["messages"][-1], "tool_calls")
        and len(state["messages"][-1].tool_calls) > 0
    ):
        tool_name = state["messages"][-1].tool_calls[0]["name"]
        if tool_name in primary_assistant_tools_names:
            # print("Primary Assistant Tool Call: ", tool_name)
            return PRIMARY_ASSISTANT_TOOLS_NODE
        # Pydantic here
        elif tool_name == HandoffToFlightDiscoveryAgent.__name__:
            # print("Primary Assistant Tool Call Flight Discovery Agent")
            return FD_BASELINE_ENTRY_NODE
        elif tool_name == HandoffToFlightSearchAgent.__name__:
            # print("Primary Assistant Tool Call Flight Search Assistant")
            return FLIGHT_SEARCH_HANDOFF_NODE
        else:
            # print("Primary Assistant ends: Unknown tool name......")
            return Command(
                goto=PRIMARY_ASSISTANT,
                update={
                    "messages": [
                        ToolMessage(
                            content="Tool call failed, tool name not found, do NOT make up tools or parameters! Continue assisting, accounting for the user's input."
                        )
                    ]
                },
            )
    else:
        # print("Primary Assistant END")
        return END


builder.add_conditional_edges(
    PRIMARY_ASSISTANT,
    decision_after_primary_thought,
    {
        PRIMARY_ASSISTANT_TOOLS_NODE: PRIMARY_ASSISTANT_TOOLS_NODE,
        # "flight_entry_node": "flight_entry_node",
        # TAXI_ENTRY_NODE: TAXI_ENTRY_NODE,
        END: END,
        FLIGHT_SEARCH_HANDOFF_NODE: FLIGHT_SEARCH_HANDOFF_NODE,
        FD_BASELINE_ENTRY_NODE: FD_BASELINE_ENTRY_NODE,
    },
)
builder.add_node(PRIMARY_ASSISTANT_TOOLS_NODE, ToolNode(primary_assistant_tools))
builder.add_edge(PRIMARY_ASSISTANT_TOOLS_NODE, PRIMARY_ASSISTANT)

# builder.add_node(INTENT_GRAPH, intent_graph)
# builder.add_edge(INTENT_GRAPH, END)


def flight_search_handoff_node(
    state: Annotated[State, InjectedState],
):
    # Log the arguments provided by ToFlightSearchAssistant
    # print(
    #     "-------------------------------- flight_search_handoff_node --------------------------------"
    # )
    tool_call = state["messages"][-1].tool_calls[0]
    tool_call_id = tool_call["id"]
    tool_args = tool_call["args"]

    task_description = tool_args.get("task_description")
    requirement_id = tool_args.get("requirement_id")

    tool_msg = ToolMessage(
        tool_call_id=tool_call_id,
        content=(
            f"Control transferred from primary assistant to flight search assistant. task_description={task_description}"
        ),
    )
    return Command(
        update={
            "messages": [tool_msg],
            "dialog_state": ["in_flight_search_assistant"],
            "requirement_id": requirement_id,
        },
    )


# Node to invoke the flight search agent
def flight_search_invoke_node(
    supervisor_state: State,
    config: RunnableConfig,
):
    # print(
    #     "-------------------------------- flight_search_invoke_node --------------------------------"
    # )
    # Get the flight search message history and add the latest user message
    flight_search_messages = supervisor_state.get("flight_search_messages", []).copy()

    # Find the latest User message from state's message list
    messages = supervisor_state.get("messages", [])
    last_user_msg = None
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            last_user_msg = m
            break

    # Add the latest user message to flight search history if it's new
    if last_user_msg and (
        not flight_search_messages
        or flight_search_messages[-1].content != last_user_msg.content
    ):
        flight_search_messages.append(last_user_msg)

    worker_input = {
        "messages": flight_search_messages,
        "requirement_id": supervisor_state.get("requirement_id"),
        "handoff": False,  # reset on entry
        "handoff_reason": None,  # reset on entry
        "remaining_steps": 24,  # initialize loop budget for worker agent
    }

    res = flight_search_graph.invoke(
        worker_input, config
    )  # own checkpointer, same thread_id

    # Extract the last assistant message from the worker to surface in supervisor transcript
    worker_msgs = res.get("messages") or []
    last_ai_msg = None
    for m in reversed(worker_msgs):
        msg_type = getattr(m, "type", None)
        if msg_type == "ai" or isinstance(m, AIMessage):
            last_ai_msg = m
            break

    updates: dict = {}

    # Update flight search message history with all messages from the worker
    updates["flight_search_messages"] = worker_msgs

    # if worker finished its task, it sends handoff flag, pop dialog mode and surface a supervisor-level message
    if res.get("handoff"):
        reason = (
            res.get("handoff_reason")
            or "Task completed. Returning control to supervisor."
        )
        handoff_msg = AIMessage(content=f"[Flight Search Agent] Handoff: {reason}")
        updates["messages"] = [handoff_msg]
        updates["dialog_state"] = ["pop"]

        return Command(
            update=updates,
            goto=PRIMARY_ASSISTANT,
        )
    else:
        if last_ai_msg is not None:
            # append worker's AIMessage to supervisor's message list with agent name flag
            # tagged_content = f"<agent>flight_search_agent</agent><message>{last_ai_msg.content}</message>"
            updates["messages"] = [AIMessage(content=last_ai_msg.content)]

        return Command(
            update=updates,
            goto=END,
        )

    # return updates


builder.add_node(FLIGHT_SEARCH_HANDOFF_NODE, flight_search_handoff_node)
builder.add_edge(FLIGHT_SEARCH_HANDOFF_NODE, FLIGHT_SEARCH_INVOKE_NODE)
builder.add_node(FLIGHT_SEARCH_INVOKE_NODE, flight_search_invoke_node)
builder.add_edge(FLIGHT_SEARCH_INVOKE_NODE, END)

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
# register_flight_graph(builder)
# register_taxi_graph(builder)
register_intent_graph(builder)
register_fd_baseline_graph(builder)
register_fd_discovery_graph(builder)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
