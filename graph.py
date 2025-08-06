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
from langchain_core.messages import ToolMessage

from primary_assistant_tools import ToFlightAssistant, ToTaxiAssistant
from intent import intent_graph
from graph_flight import register_flight_graph
from graph_taxi import register_taxi_graph
from tools_taxi import fetch_user_taxi_requests
from graph_setup import (
    FETCH_USER_INFO_NODE,
    PRIMARY_ASSISTANT,
    PRIMARY_ASSISTANT_TOOLS_NODE,
    TAXI_ENTRY_NODE,
    TAXI_ASSISTANT,
    INTENT_GRAPH,
)

builder = StateGraph(State)


def fetch_user_info_node(state: State, config: RunnableConfig):
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured, cannot fetch user info.")
    user_flight_info = fetch_user_flight_information.invoke({})
    user_taxi_info = fetch_user_taxi_requests.invoke({})
    user_info = {
        "flight_info": user_flight_info,
        "taxi_info": user_taxi_info,
    }
    formatted_user_info = json.dumps(user_info, indent=2)
    # return {"user_info": formatted_user_info}  # user_info is a string
    return {"user_flight_info": user_flight_info, "user_taxi_info": user_taxi_info}



# ------ "sub-graph" for primary assistant ------
builder.add_node(FETCH_USER_INFO_NODE, fetch_user_info_node)
builder.add_edge(START, FETCH_USER_INFO_NODE)
def decision_after_fetch_user_info(state: State, config: RunnableConfig):
    dialog_state_list = state.get("dialog_state")
    if not dialog_state_list:
        print("No dialog state, to Primary Assistant")
        return PRIMARY_ASSISTANT
    dialog_state = dialog_state_list[-1]

    if dialog_state == "in_flight_assistant":
        print("--- shortcut to Flight Assistant ---")
        return "flight_assistant"
    elif dialog_state == "in_taxi_assistant":
        print("--- shortcut to Taxi Assistant ---")
        return TAXI_ASSISTANT
    elif dialog_state == "in_intent_elicitation_assistant":
        print("--- shortcut to Intent Elicitation Assistant ---")
        return INTENT_GRAPH
    else:
        return PRIMARY_ASSISTANT

builder.add_conditional_edges(
    FETCH_USER_INFO_NODE,
    decision_after_fetch_user_info,
    {
        PRIMARY_ASSISTANT: PRIMARY_ASSISTANT,
        TAXI_ASSISTANT: TAXI_ASSISTANT,
        INTENT_GRAPH: INTENT_GRAPH,
        "flight_assistant": "flight_assistant",
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
            print("Primary Assistant Tool Call: ", tool_name)
            return PRIMARY_ASSISTANT_TOOLS_NODE
        # Pydantic here
        elif tool_name == ToFlightAssistant.__name__:
            print("Primary Assistant Tool Call Flight Assistant")
            return "flight_entry_node"
        elif tool_name == ToTaxiAssistant.__name__:
            print("Primary Assistant Tool Call Taxi Assistant")
            return TAXI_ENTRY_NODE
        else:
            print("Primary Assistant ends: Unknown tool name......")
            return Command(
                goto=PRIMARY_ASSISTANT,
                update={
                    "messages": [
                        ToolMessage(content="Tool call failed, tool name not found, do NOT make up tools or parameters! Continue assisting, accounting for the user's input.")
                    ]
                },
            )
    else:
        print("Primary Assistant END")
        return END


builder.add_conditional_edges(
    PRIMARY_ASSISTANT,
    decision_after_primary_thought,
    {
        PRIMARY_ASSISTANT_TOOLS_NODE: PRIMARY_ASSISTANT_TOOLS_NODE,
        "flight_entry_node": "flight_entry_node",
        TAXI_ENTRY_NODE: TAXI_ENTRY_NODE,
        END: END,
    },
)
builder.add_node(PRIMARY_ASSISTANT_TOOLS_NODE, ToolNode(primary_assistant_tools))

builder.add_edge(PRIMARY_ASSISTANT_TOOLS_NODE, PRIMARY_ASSISTANT)

builder.add_node(INTENT_GRAPH, intent_graph)
builder.add_edge(INTENT_GRAPH, END)

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
register_flight_graph(builder)
register_taxi_graph(builder)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")