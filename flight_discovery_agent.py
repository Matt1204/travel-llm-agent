from langgraph.graph import StateGraph, START, END
from fd_baseline_chain import flight_discovery_chain, baseline_tools
from primary_assistant_chain import Assistant, State
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver
import uuid
from langgraph.checkpoint.mongodb import MongoDBSaver
import os
from dotenv import load_dotenv
from custom_tool_node import create_simple_tool_node
from utils import chat_bot_print



from primary_assistant_tools import WorkerCompleteOrEscalate
from graph_setup import (
    FD_BASELINE_ENTRY_NODE,
    FD_BASELINE_AGENT,
    FD_BASELINE_TOOL_HANDLER,
    FD_BASELINE_TOOL_NODE,
    FD_DISCOVERY_AGENT,
    PRIMARY_ASSISTANT,
    FD_RETURN_NODE,
    FD_DISCOVERY_ENTRY_NODE,
    FD_DISCOVERY_TOOL_NODE,
    FD_DISCOVERY_TOOL_HANDLER,
)
from intent import load_flight_requirements_from_db
from fd_discovery_chain import FDSnapshot, fd_discovery_chain, fd_discovery_tools
from fd_discovery_graph import register_fd_discovery_graph
from db_connection import db_file
import sqlite3

# CREATE TABLE "flight_discovery" (
#   "session_id" text NOT NULL,
#   "deal_id" text NOT NULL,
#   "source_requirement_id" text NOT NULL,
#   "baseline_filter" TEXT,
#   "baseline_flight" TEXT,
#   "discovery_filter" TEXT,
#   "discovery_flights" TEXT,
#   "altered_filter_val" TEXT,
#   "is_better_deal" integer,
#   PRIMARY KEY ("session_id", "deal_id")
# );

def find_discovered_deals_by_requirement_id(requirement_id: str):
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            "SELECT session_id, deal_id FROM flight_discovery WHERE source_requirement_id = ?",
            (requirement_id,),
        )
        deals = cur.fetchall()
        return deals
    except Exception as e:
        print(f"Error finding discovered deals by requirement id: {e}")
        return []


def fd_baseline_entry_node(state: State, config: RunnableConfig):
    # print(" Flight Discovery Entry Node")
    chat_bot_print(
        "Flight Discovery has begun. \n"
        "Starting searching for a flight as benchmark.",
        is_human_msg=False,
    )

    tc_message = []
    flight_discovery_entry_message = []
    requirement_id = state.get("requirement_id", None)

    last_msg = state["messages"][-1] if state["messages"] else None
    if isinstance(last_msg, AIMessage) and len(last_msg.tool_calls):
        tc = last_msg.tool_calls[0]
        args = tc.get("args") or {}
        requirement_id = args.get("requirement_id") or None
        tc_message.append(
            ToolMessage(
                tool_call_id=tc["id"],
                content="Control tranferred from primary assistant to flight_discovery_assistant.",
            )
        )

    # preparing the baseline FlightRequirements object
    # requirement_id: str | None = state.get("requirement_id", None)
    if not requirement_id:
        # No requirement id supplied – nothing to do (let supervisor handle).
        print(
            "Error: No requirement id supplied – nothing to do (let supervisor handle)."
        )
        return

    existing_deals = find_discovered_deals_by_requirement_id(requirement_id)
    if len(existing_deals) > 0:
        session_id = existing_deals[0][0]
        deal_count = len(existing_deals)
        print(
            f"Found {len(existing_deals)} existing deals for requirement id: {requirement_id}"
        )
        flight_discovery_entry_message.append(
            HumanMessage(
                content=f"Flight discovery Session associated with current flight requirement:{requirement_id} has been performed before; found under discovery session_id:{session_id} with {deal_count} deals found. call 'WorkerCompleteOrEscalate' to stop current session and handoff control back to primary assistant.",
            )
        )
        return {
            "messages": tc_message,
            "flight_discovery_messages": flight_discovery_entry_message,
            "dialog_state": ["in_flight_discovery_assistant"],
            "flight_requirements": None,
        }
    else:
        flight_discovery_entry_message.append(
            HumanMessage(
                content=f"Start flight discovery process for requirement_id: {requirement_id}",
            )
        )

    req_obj = load_flight_requirements_from_db(requirement_id)
    # The calling graph will merge this into overall state.

    return {
        "messages": tc_message,
        "flight_discovery_messages": flight_discovery_entry_message,
        "flight_requirements": req_obj,
        "dialog_state": ["in_flight_discovery_assistant"],
        "requirement_id": requirement_id,
    }



def baseline_tool_handler(state: State, config: RunnableConfig) -> Command[
    Literal[
        FD_BASELINE_TOOL_NODE,
        FD_BASELINE_AGENT,
        FD_RETURN_NODE,
        END,
    ]
]:
    # print(
    #     f" ------------------ baseline_tool_handler --------------------------------"
    # )
    go_to = tools_condition(state, messages_key="flight_discovery_messages")
    if go_to == END:
        return Command(goto=END)

    tool_call_list = state["flight_discovery_messages"][-1].tool_calls or []
    tool_name_list = [tc["name"] for tc in tool_call_list]
    if tool_name_list:
        if WorkerCompleteOrEscalate.__name__ in tool_name_list:
            return Command(
                goto=FD_RETURN_NODE
            )  # if Job done or need help, return to primary assistant
        else:
            return Command(goto=FD_BASELINE_TOOL_NODE)  # toute to ToolNode
    else:
        return Command(goto=FD_BASELINE_AGENT)


def fd_return_node(state: State, config: RunnableConfig):
    print("HITTTTTTT Flight Discovery Return Node")
    tool_massage = []
    return_message = []
    reason = ""

    # WorkerCompleteOrEscalate called, append a ToolMessage
    if state["flight_discovery_messages"][-1].tool_calls:
        reason = state["flight_discovery_messages"][-1].tool_calls[0]["args"]["reason"]
        tool_massage.append(
            ToolMessage(
                content="Handing off to the Primary Assistant from Flight Discovery Assistant. Reason: "
                + reason,
                tool_call_id=state["flight_discovery_messages"][-1].tool_calls[0]["id"],
            )
        )

        return_message = HumanMessage(
            content="Flight Discovery Agent handoff control back to Primary Assistant, Reason: "
            + reason
            + "\nReflect on the past conversation and keep assisting the user as needed.",
        )

    return {
        "messages": return_message,
        "flight_discovery_messages": tool_massage,
        "requirement_id": "",
        "dialog_state": ["pop"],
    }


def register_fd_baseline_graph(builder: StateGraph):
    builder.add_node(FD_BASELINE_ENTRY_NODE, fd_baseline_entry_node)
    # builder.add_edge(START, FD_BASELINE_ENTRY_NODE)

    builder.add_node(
        FD_BASELINE_AGENT,
        Assistant(flight_discovery_chain, messages_key="flight_discovery_messages"),
    )
    builder.add_edge(FD_BASELINE_ENTRY_NODE, FD_BASELINE_AGENT)

    builder.add_node(FD_BASELINE_TOOL_HANDLER, baseline_tool_handler)
    builder.add_edge(FD_BASELINE_AGENT, FD_BASELINE_TOOL_HANDLER)

    builder.add_node(
        FD_BASELINE_TOOL_NODE,
        create_simple_tool_node(
            baseline_tools, messages_key="flight_discovery_messages"
        ),
    )
    builder.add_edge(FD_BASELINE_TOOL_NODE, FD_BASELINE_AGENT)

    builder.add_node(FD_RETURN_NODE, fd_return_node)
    # builder.add_edge(FD_RETURN_NODE, PRIMARY_ASSISTANT)
    builder.add_edge(
        FD_RETURN_NODE, PRIMARY_ASSISTANT
    )  # TODO: make this go to primary assistant in the future !!!!!!!!!!!!!!!!


def find_ckpt_before_node(graph, thread_id: str, node_name: str) -> str:
    """Return checkpoint_id whose 'next' includes node_name (newest-first search)."""
    cfg = {"configurable": {"thread_id": thread_id}}
    for snap in graph.get_state_history(cfg):
        next_nodes = tuple(snap.next or ())
        for next in next_nodes:
            if node_name in next:
                print(
                    f"Found checkpoint_id: {snap.config['configurable']['checkpoint_id']}"
                )
                return snap.config["configurable"]["checkpoint_id"]
    raise RuntimeError(f"No checkpoint found where next includes {node_name!r}")


if __name__ == "__main__":
    with MongoDBSaver.from_conn_string(
        os.getenv("MONGODB_CONNECTION_STR")
    ) as checkpointer:
        graph_builder = StateGraph(State)
        register_fd_baseline_graph(graph_builder)
        register_fd_discovery_graph(graph_builder)
        flight_discovery_graph = graph_builder.compile(checkpointer=checkpointer)

        my_thread_id = str(uuid.uuid4())
        while True:
            q = input("debug input: ")
            if q == "exit":
                break

                # ---------- Discovery Agent ----------
                # resume_cfg = {
                #     "configurable": {
                #         "thread_id": "a99ee401-fa6b-4987-ba1b-a0bc607a4129",
                #         "checkpoint_id": "1f07c599-d13f-68b6-800a-33c6707384c4",
                #     },
                #     "recursion_limit": 100,
                # }
                # # This will execute fd_discovery_entry_node first
                # stream = flight_discovery_graph.stream(
                #     None, resume_cfg, stream_mode=["values"]
                # )
                # for event in stream:
                event[-1]["flight_discovery_messages"][-1].pretty_print()

            # ---------- Baseline Agent ----------
            config = {
                "configurable": {
                    "passenger_id": "3442 587242",
                    "thread_id": my_thread_id,
                },
                "recursion_limit": 100,
            }
            stream = flight_discovery_graph.stream(
                {
                    "messages": [HumanMessage(content=q)],
                    "flight_discovery_messages": [HumanMessage(content=q)],
                    "requirement_id": "69e50f15-4d3f-4a00-9f1b-b2cd363c7300",
                },
                config,
                stream_mode=["values"],
            )
            for event in stream:
                event[-1]["flight_discovery_messages"][-1].pretty_print()
            cur_state = flight_discovery_graph.get_state(config)

        ckpt_id = find_ckpt_before_node(
            flight_discovery_graph, my_thread_id, FD_DISCOVERY_ENTRY_NODE
        )
        print(f"Thread ID: {my_thread_id}, Checkpoint ID: {ckpt_id} !!!!!!!!!!!!!!!")
