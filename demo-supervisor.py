from __future__ import annotations

from typing import Annotated, Literal, List, TypedDict, Optional
from operator import add
from datetime import datetime
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from flight_search_agent import flight_search_graph
# Note: we only need the flight worker's state for building its subgraph in the demo


# ---------------------------------------------------------------------------
# 1) Supervisor state: different from workers' state (no shared `messages`)
# ---------------------------------------------------------------------------
# - The supervisor keeps its own message log (`super_messages`).
# - A simple `dialog_mode` stack controls which worker is "active".
# - We DO NOT put the worker's messages here. Workers keep their own `messages`
#   inside their own subgraphs/checkpoint namespaces.
# ---------------------------------------------------------------------------


def merge_user_msg_list(prev: list, value: list):
    return (prev or []) + (value or [])


def update_dialog_stack(prev_val: list[str], value: list[str]):
    """Push-on-update; pop when we get ["pop"]."""
    if value == ["pop"]:
        return (prev_val or [])[:-1]
    return (prev_val or []) + value


class SupervisorState(TypedDict):
    super_messages: Annotated[list[AnyMessage], add]  # supervisor's chat history
    dialog_mode: Annotated[
        List[Literal["in_flight", "in_primary"]], update_dialog_stack
    ]
    # optional typed fields you want to pass down
    requirement_id: Optional[str]
    user_msg_list: Annotated[
        List[str], merge_user_msg_list
    ]  # capture last user text(s)


# ---------------------------------------------------------------------------
# 2) Simple supervisor node
# ---------------------------------------------------------------------------
def supervisor_node(state: SupervisorState, config: RunnableConfig):
    msgs = state.get("super_messages", [])
    last_user_text = (state.get("user_msg_list") or [""])[-1].lower()

    # Naive routing rule for demo: anything mentioning "flight" -> the flight agent
    if "flight" in last_user_text or "ticket" in last_user_text:
        return Command(
            goto="flight_entry",  # jump into the worker wrapper
            update={"dialog_mode": ["in_flight"]},  # push worker mode
        )

    # Otherwise, respond as supervisor (minimal demo response)
    # In production: call an LLM and tools here.
    reply = f"DUMMY RESPONSE: [Supervisor] You said: {last_user_text!r} at {datetime.now().isoformat()}"
    return {"super_messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------------------
# 3) Router at graph START: if we’re in a worker mode, bypass supervisor
# ---------------------------------------------------------------------------
def entry_router(state: SupervisorState, config: RunnableConfig):
    cur_mode = state.get("dialog_mode") or []
    if cur_mode and cur_mode[-1] == "in_flight":
        return "flight_entry"
    return "supervisor"


# ---------------------------------------------------------------------------
# 4) Worker wrapper node: invoke the flight subgraph in its own namespace
# ---------------------------------------------------------------------------
# Key points:
# - We treat the FLIGHT subgraph as a node via `add_node("flight", flight_graph)`.
#   That lets HITL interrupts bubble up correctly and persist with the parent.
# - We set a dedicated checkpoint namespace for the flight subgraph by using
#   `.with_config(configurable={"checkpoint_ns": "flight"})` — this keeps
#   the worker’s messages/history separate from the supervisor’s.
# - We pass structured fields like `requirement_id` as needed; we do NOT pass
#   the supervisor's message log.
# ---------------------------------------------------------------------------


def make_flight_entry_node(flight_graph):
    """Returns a node function that dispatches to the flight subgraph."""

    # Configure the subgraph to use its own checkpoint namespace (memory silo)

    flight_node = flight_graph.with_config(
        configurable={"checkpoint_ns": "flight_namespace"}
    )
    # [UNSURE] If your LangGraph version doesn't propagate per-node config via with_config,
    # wrap `flight_node.invoke(...)` inside this function and pass the config explicitly.
    # This works in recent versions; older versions may require a manual wrapper.

    def flight_entry(state: SupervisorState, config):
        # First-time call vs subsequent calls:
        # - If the worker uses regular "messages" input, send the latest user text.
        # - If the worker is *waiting at an interrupt*, we should resume with Command(resume=...).
        #   BUT the parent node can't know for sure without inspecting subgraph state.
        #
        # The clean pattern is to *let the subgraph raise/propagate interrupt* to the parent graph.
        # Then your app resumes by calling parent_graph.invoke(Command(resume=...)).
        # We just forward the user's text into the subgraph on first hop.
        # [UNSURE] If your worker *requires* `Command(resume=...)`, add logic to detect/branch.
        last_user_msg = (state.get("user_msg_list") or [""])[-1]

        # Minimal input mapping: many worker agents expect {"messages": [("user", ...)]}
        # Handoff flags are reset to avoid stale values in the worker state
        worker_input = {
            # you can pass requirement_id down if your worker loads by ID
            "messages": [("user", last_user_msg)] if last_user_msg else [],
            "requirement_id": state.get("requirement_id"),
            "handoff": False,  # reset on entry
            "handoff_reason": None,  # reset on entry
            "remaining_steps": 24,  # initialize loop budget for worker agent
        }

        # res = flight_node.invoke(worker_input, cfg)
        res = flight_search_graph.invoke(worker_input, cfg) # own checkpointer, same thread_id

        # Extract the last assistant message from the worker to surface in supervisor transcript
        worker_msgs = res.get("messages") or []
        last_ai_msg = None
        for m in reversed(worker_msgs):
            msg_type = getattr(m, "type", None)
            if msg_type == "ai" or isinstance(m, AIMessage):
                last_ai_msg = m
                break

        updates: dict = {}
        if last_ai_msg is not None:
            # append worker's AIMessage to supervisor's message list with agent name flag
            tagged_content = f"<agent>flight_search_agent</agent><message>{last_ai_msg.content}</message>"
            updates["super_messages"] = [AIMessage(content=tagged_content)]

        # if worker finished its task, it sends handoff flag, pop dialog mode and surface a supervisor-level message
        if res.get("handoff"):
            reason = (
                res.get("handoff_reason")
                or "Task completed. Returning control to supervisor."
            )
            handoff_msg = AIMessage(content=f"[Flight Agent] Handoff: {reason}")
            if "super_messages" in updates:
                updates["super_messages"].append(handoff_msg)
            else:
                updates["super_messages"] = [handoff_msg]
            updates["dialog_mode"] = ["pop"]

        return updates

    return flight_entry


# ---------------------------------------------------------------------------
# 5) Assembly helper
# ---------------------------------------------------------------------------


def build_supervisor_graph(flight_graph):
    """Create a parent graph with a supervisor and a flight subgraph worker."""
    builder = StateGraph(SupervisorState)

    # Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("flight_entry", make_flight_entry_node(flight_graph))

    # Top-level routing (bypass while in worker mode)
    builder.add_conditional_edges(
        START,
        entry_router,
        {
            "supervisor": "supervisor",
            "flight_entry": "flight_entry",
        },
    )

    # After supervisor runs, go back to START to route again next turn
    builder.add_edge("supervisor", END)
    builder.add_edge("flight_entry", END)

    return builder


# ---------------------------------------------------------------------------
# 6) Demo: compile & run
# ---------------------------------------------------------------------------
# You can swap InMemorySaver with Postgres/Redis/etc in production.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # [UNSURE] Replace this with your real compiled flight subgraph.
    # It must accept {"messages": [...] } and/or {"requirement_id": "..."}.
    from flight_search_agent import (
        register_flight_search_graph,
        FlightSearchState,
    )  # your worker

    worker_builder = StateGraph(FlightSearchState)  # worker defines its own schema
    register_flight_search_graph(worker_builder)
    flight_graph = worker_builder.compile(checkpointer=InMemorySaver())

    # Build supervisor app
    supervisor_builder = build_supervisor_graph(flight_graph)
    app = supervisor_builder.compile(checkpointer=InMemorySaver())

    # One thread per end-user session
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break

        input_state = {
            "super_messages": [HumanMessage(content=user_input)],
            "user_msg_list": [user_input],
            "requirement_id": "73e020b5-0a2f-4320-92c0-24de6fa3fd97",
        }

        events = app.stream(input_state, cfg, stream_mode="values")
        for ev in events:
            ev["super_messages"][-1].pretty_print()
        cur_super_state = app.get_state(cfg)
        print("--------")
