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


# TODO:
# 1. in the Hand-off tool, ask supervisor to provide flight_req_id


def fd_baseline_entry_node(state: State, config: RunnableConfig):
    print("HITTTTTTT Flight Discovery Entry Node")

    tc_message = []
    flight_discovery_entry_message = []

    last_msg = state["messages"][-1] if state["messages"] else None
    if isinstance(last_msg, ToolMessage) and len(last_msg.tool_calls):
        tc = last_msg.tool_calls[0]
        args = tc.get("args") or {}
        request = args.get("request")
        tc_message.append(
            ToolMessage(
                tool_call_id=tc["id"],
                content="Control tranferred from primary assistant to flight_discovery_assistant.",
            )
        )

    flight_discovery_entry_message.append(
        HumanMessage(
            content="Control tranferred from primary assistant to flight_discovery_assistant. Start flight discovery process.",
        )
    )

    # preparing the baseline FlightRequirements object
    requirement_id: str | None = state.get("requirement_id", None)
    if not requirement_id:
        # No requirement id supplied – nothing to do (let supervisor handle).
        print("No requirement id supplied – nothing to do (let supervisor handle).")
        return
    req_obj = load_flight_requirements_from_db(requirement_id)
    # The calling graph will merge this into overall state.

    return {
        "messages": tc_message,
        "flight_discovery_messages": flight_discovery_entry_message,
        "flight_requirements": req_obj,
        "dialog_state": ["in_flight_discovery_assistant"],
    }


def baseline_tool_handler(state: State, config: RunnableConfig) -> Command[
    Literal[
        FD_BASELINE_TOOL_NODE,
        FD_BASELINE_AGENT,
        FD_RETURN_NODE,
        END,
    ]
]:
    print(
        f" ------------------ 🔍 baseline_tool_handler --------------------------------"
    )
    go_to = tools_condition(state, messages_key="flight_discovery_messages")
    print(f"🔍 tools_condition returned: {go_to}")
    if go_to == END:
        print(f"🔍 baseline_tool_handler: Going to END")
        return Command(goto=END)

    tool_call_list = state["flight_discovery_messages"][-1].tool_calls or []
    tool_name_list = [tc["name"] for tc in tool_call_list]
    print(f"🔍 baseline_tool_handler: Found tools: {tool_name_list}")
    if tool_name_list:
        if WorkerCompleteOrEscalate.__name__ in tool_name_list:
            print(
                f"🔍 baseline_tool_handler: Going to FD_RETURN_NODE (WorkerCompleteOrEscalate)"
            )
            return Command(
                goto=FD_RETURN_NODE
            )  # if Job done or need help, return to primary assistant
        else:
            print(f"🔍 baseline_tool_handler: Going to FD_BASELINE_TOOL_NODE")
            return Command(goto=FD_BASELINE_TOOL_NODE)  # toute to ToolNode
    else:
        print(f"🔍 baseline_tool_handler: No tools found, going to FD_BASELINE_AGENT")
        return Command(goto=FD_BASELINE_AGENT)


def fd_return_node(state: State, config: RunnableConfig):
    print("HITTTTTTT Flight Discovery Return Node")
    tool_massage = []

    # WorkerCompleteOrEscalate called, append a ToolMessage
    if state["flight_discovery_messages"][-1].tool_calls:
        tool_massage.append(
            ToolMessage(
                content="Handing off to the Primary Assistant from Flight Discovery Assistant. Reason: "
                + state["flight_discovery_messages"][-1].tool_calls[0]["args"][
                    "reason"
                ],
                tool_call_id=state["flight_discovery_messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "messages": tool_massage,
        "flight_discovery_messages": tool_massage,
        "dialog_state": ["pop"],
    }


# ------------------------------------------------------------
# Flight Discovery Graph
# ------------------------------------------------------------
def fd_discovery_entry_node(
    state: State, config: RunnableConfig
) -> Command[Literal[FD_BASELINE_ENTRY_NODE, FD_DISCOVERY_AGENT]]:
    print(
        "-------------------------------- HITTTTTTT fd_discovery_entry_node --------------------------------"
    )
    session_info = state.get("fd_session_info") or {}
    baseline = session_info.get("baseline") or {}
    system_discovered = session_info.get("system_discovered") or []
    if not baseline:
        print("Error:No baseline flight found, going back to baseline entry node")
        return Command(goto=FD_BASELINE_ENTRY_NODE)
    else:
        # if this is just come from baseline entry node, create a new FDSnapshot as current flight snapshot
        if len(system_discovered) == 0:
            cur_snapshot = FDSnapshot(
                id=str(uuid.uuid4()),
                filter_used=None,
                flights=None,
                comment=None,
            )
            system_discovered.append(cur_snapshot)
            session_info["system_discovered"] = system_discovered
            print("initializing system_discovered list with a new snapshot")

            # Clear prior discovery chat but keep the latest ToolMessage if present
            human_message = HumanMessage(
                content="Control tranferred from baseline agent to discovery agent. Start flight discovery process.",
            )

            updates = {"fd_session_info": session_info}
            updates["flight_discovery_messages"] = ["__RESET__", human_message]

            return Command(update=updates)
        else:
            cur_snapshot = system_discovered[-1]
            if (
                cur_snapshot.get("flights") is not None
                and cur_snapshot.get("filter_used") is not None
                and cur_snapshot.get("comment") is not None
            ):
                new_snapshot = FDSnapshot(
                    id=str(uuid.uuid4()),
                    filter_used=None,
                    flights=None,
                    comment=None,
                )
                system_discovered.append(new_snapshot)
                session_info["system_discovered"] = system_discovered
                print(
                    "current snapshot has all fields(flights, filter_used, comment), 2 steps done. Append a new snapshot, move on to next round of discovery."
                )
                return Command(
                    goto=FD_DISCOVERY_AGENT, update={"fd_session_info": session_info}
                )

            if (
                cur_snapshot.get("flights") is not None
                and cur_snapshot.get("filter_used") is not None
                and cur_snapshot.get("comment") is None
            ):
                print(
                    "current snapshot has fields(flights, filter_used), missing comment. Discovery step done. Move on to evaluation step."
                )
                return Command(goto=FD_DISCOVERY_AGENT)
            else:
                print("Fall back. un-handled current snapshot values.")
                return Command(goto=FD_DISCOVERY_AGENT)


def fd_discovery_tool_handler(state: State, config: RunnableConfig) -> Command[
    Literal[
        FD_DISCOVERY_TOOL_NODE,
        FD_DISCOVERY_AGENT,
        FD_RETURN_NODE,
        END,
    ]
]:
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
            return Command(goto=FD_DISCOVERY_TOOL_NODE)  # toute to ToolNode
    else:
        print("Tool Handler: no tool call found, going to END")
        return Command(goto=END)


def register_fd_baseline_graph(builder: StateGraph):
    builder.add_node(FD_BASELINE_ENTRY_NODE, fd_baseline_entry_node)
    builder.add_edge(START, FD_BASELINE_ENTRY_NODE)

    builder.add_node(
        FD_BASELINE_AGENT,
        Assistant(flight_discovery_chain, messages_key="flight_discovery_messages"),
    )
    builder.add_edge(FD_BASELINE_ENTRY_NODE, FD_BASELINE_AGENT)

    builder.add_node(FD_BASELINE_TOOL_HANDLER, baseline_tool_handler)
    builder.add_edge(FD_BASELINE_AGENT, FD_BASELINE_TOOL_HANDLER)

    def debug_tool_node(state, config):
        print(
            f" ------------------ 🔥 FD_BASELINE_TOOL_NODE called with state keys: {list(state.keys())}"
        )
        tool_call_count = len(state.get("flight_discovery_messages", []))
        print(f"🔥 Number of flight_discovery_messages: {tool_call_count}")
        if state.get("flight_discovery_messages"):
            last_msg = state["flight_discovery_messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(
                    f"🔥 Tool calls in last message: {[tc['name'] for tc in last_msg.tool_calls]}"
                )
                print(f"🔥 Tool call details: {last_msg.tool_calls}")

        # Check if the state has the required structure for ToolNode
        print(f"🔥 State structure check:")
        print(
            f"   - flight_discovery_messages type: {type(state.get('flight_discovery_messages'))}"
        )
        print(
            f"   - flight_discovery_messages length: {len(state.get('flight_discovery_messages', []))}"
        )

        # Try custom tool executor first, fallback to ToolNode if it fails

        try:
            print(f"🔥 Falling back to ToolNode.invoke...")
            tool_node = ToolNode(
                baseline_tools,
                messages_key="flight_discovery_messages",
                handle_tool_errors=True,
            )
            print(f"🔥 ToolNode created successfully")
            print(f"🔥 Calling tool_node.invoke with config: {config}")
            result = tool_node.invoke(state, config)
            print(f"🔥 ToolNode.invoke completed successfully!")
            print(
                f"🔥 ToolNode result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}"
            )
            return result
        except Exception as toolnode_error:
            print(f"🔥 ERROR in ToolNode.invoke: {toolnode_error}")
            import traceback

            traceback.print_exc()
            # Return a fallback result to prevent complete failure
            from langchain_core.messages import ToolMessage

            return {
                "flight_discovery_messages": [
                    ToolMessage(
                        content=f"Tool execution failed: {toolnode_error}",
                        tool_call_id="error",
                    )
                ]
            }

    # builder.add_node(FD_BASELINE_TOOL_NODE, create_simple_tool_node(baseline_tools, messages_key="flight_discovery_messages"))
    # builder.add_node(
    #     FD_BASELINE_TOOL_NODE,
    #     ToolNode(
    #         baseline_tools,
    #         messages_key="flight_discovery_messages",
    #         handle_tool_errors=True,
    #     ),
    # )
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
        FD_RETURN_NODE, END
    )  # TODO: make this go to primary assistant in the future !!!!!!!!!!!!!!!!


def register_fd_discovery_graph(builder: StateGraph):
    builder.add_node(FD_DISCOVERY_ENTRY_NODE, fd_discovery_entry_node)
    # builder.add_edge(START, FD_DISCOVERY_ENTRY_NODE)

    builder.add_node(
        FD_DISCOVERY_AGENT,
        Assistant(fd_discovery_chain, messages_key="flight_discovery_messages"),
    )
    builder.add_edge(FD_DISCOVERY_ENTRY_NODE, FD_DISCOVERY_AGENT)
    # builder.add_edge(FD_DISCOVERY_ENTRY_NODE, END)

    builder.add_node(FD_DISCOVERY_TOOL_HANDLER, fd_discovery_tool_handler)
    builder.add_edge(FD_DISCOVERY_AGENT, FD_DISCOVERY_TOOL_HANDLER)

    builder.add_node(
        FD_DISCOVERY_TOOL_NODE,
        ToolNode(fd_discovery_tools, messages_key="flight_discovery_messages"),
    )
    builder.add_edge(FD_DISCOVERY_TOOL_NODE, FD_DISCOVERY_ENTRY_NODE)


def find_ckpt_before_node(graph, thread_id: str, node_name: str) -> str:
    """Return checkpoint_id whose 'next' includes node_name (newest-first search)."""
    cfg = {"configurable": {"thread_id": thread_id}}
    for snap in graph.get_state_history(cfg):
        nxt = tuple(snap.next or ())
        if node_name in nxt:
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

        my_thread_id = "0815-15-thread-id"
        config = {
            "configurable": {
                "passenger_id": "3442 587242",
                "thread_id": my_thread_id,
            }
        }

        while True:
            q = input("debug input: ")
            if q == "exit":
                break

            # Discovery Agent
            resume_cfg = {
                "configurable": {
                    "thread_id": my_thread_id,
                    "checkpoint_id": "1f07a130-ee7f-6de6-800a-16162498243f",
                },
                "recursion_limit": 100,
            }
            # This will execute fd_discovery_entry_node first
            stream = flight_discovery_graph.stream(
                None, resume_cfg, stream_mode=["values"]
            )
            for event in stream:
                event[-1]["flight_discovery_messages"][-1].pretty_print()

            # Baseline Agent
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
        print(f"Checkpoint ID: {ckpt_id}")
