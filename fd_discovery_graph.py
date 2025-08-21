from fd_discovery_chain import fd_discovery_chain, fd_discovery_tools
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from typing import Literal, Dict, Any
import uuid
import sqlite3
import json

from primary_assistant_tools import WorkerCompleteOrEscalate
from primary_assistant_chain import State, Assistant, FDSnapshot
from graph_setup import (
    FD_DISCOVERY_ENTRY_NODE,
    FD_DISCOVERY_AGENT,
    FD_DISCOVERY_TOOL_NODE,
    FD_DISCOVERY_TOOL_HANDLER,
    FD_RETURN_NODE,
    FD_BASELINE_ENTRY_NODE,
)
from db_connection import db_file

CONTINUE_DISCOVERY = "continue_discovery"
END_DISCOVERY = "end_discovery"


def review_discovery_session(state: State):
    session_info = state.get("fd_session_info") or {}
    system_discovered = session_info.get("system_discovered") or []
    total_attempts = len(system_discovered)

    successful_attempts = [
        snapshot for snapshot in system_discovered if snapshot.get("is_better_deal")
    ]
    count_successful_attempts = len(successful_attempts)
    failed_attempts = [
        snapshot for snapshot in system_discovered if not snapshot.get("is_better_deal")
    ]
    count_failed_attempts = len(failed_attempts)
    print(
        f"Total attempts: {total_attempts}, Successful attempts: {count_successful_attempts}, Failed attempts: {count_failed_attempts}"
    )

    if count_successful_attempts >= 3 or count_failed_attempts >= 6:
        return END_DISCOVERY
    else:
        return CONTINUE_DISCOVERY


def save_discovery_session(state: State):
    session_info = state.get("fd_session_info") or {}
    system_discovered = session_info.get("system_discovered") or []
    session_id = str(uuid.uuid4())
    source_requirement_id = state.get("requirement_id") or ""
    baseline = session_info.get("baseline") or {}
    baseline_filter = (
        (baseline.get("filter_used") or {}) if isinstance(baseline, dict) else {}
    )
    baseline_flights = (
        (baseline.get("flights") or []) if isinstance(baseline, dict) else []
    )

    def _compute_altered_values(
        base_filter: Dict[str, Any], current_filter: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not isinstance(base_filter, dict):
            base_filter = {}
        if not isinstance(current_filter, dict):
            current_filter = {}
        altered: Dict[str, Any] = {}
        keys = set(base_filter.keys()) | set(current_filter.keys())
        for key in keys:
            if key not in current_filter:
                continue
            cur_val = current_filter.get(key)
            base_val = base_filter.get(key)
            if isinstance(cur_val, list) and isinstance(base_val, list):
                if cur_val != base_val:
                    altered[key] = cur_val
            else:
                if cur_val != base_val:
                    altered[key] = cur_val
        return altered

    def _to_jsonable(value):
        try:
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
        except Exception:
            pass
        from datetime import datetime as _dt

        if isinstance(value, _dt):
            try:
                return value.isoformat(sep=" ")
            except Exception:
                return str(value)
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_jsonable(v) for v in list(value)]
        return value

    # successful_attempts = [
    #     snapshot for snapshot in system_discovered if snapshot.get("is_better_deal")
    # ]
    # if not successful_attempts:
    #     print("save_discovery_session: No successful attempts to persist.")
    #     return

    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()

        baseline_filter_json = json.dumps(
            _to_jsonable(baseline_filter), ensure_ascii=False
        )
        baseline_flight_obj = (
            baseline_flights[0]
            if isinstance(baseline_flights, list) and baseline_flights
            else None
        )
        baseline_flight_json = json.dumps(
            _to_jsonable(baseline_flight_obj), ensure_ascii=False
        )

        for snapshot in system_discovered:
            is_better_deal = 1 if snapshot.get("is_better_deal") else 0
            discovery_filter = snapshot.get("filter_used") or {}
            altered = (
                _compute_altered_values(baseline_filter, discovery_filter)
                if isinstance(discovery_filter, dict)
                else {}
            )
            discovery_flights = snapshot.get("flights") or []

            # Build JSON payloads
            discovery_filter_json = json.dumps(
                _to_jsonable(discovery_filter), ensure_ascii=False
            )
            discovery_flights_json = json.dumps(
                _to_jsonable(discovery_flights), ensure_ascii=False
            )
            altered_json = json.dumps(_to_jsonable(altered), ensure_ascii=False)

            deal_id = snapshot.get("id") or str(uuid.uuid4())

            cur.execute(
                (
                    "INSERT OR REPLACE INTO flight_discovery (session_id, deal_id, source_requirement_id, baseline_filter, baseline_flight, discovery_filter, discovery_flights, altered_filter_val, is_better_deal) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    session_id,
                    deal_id,
                    source_requirement_id,
                    baseline_filter_json,
                    baseline_flight_json,
                    discovery_filter_json,
                    discovery_flights_json,
                    altered_json,
                    is_better_deal,
                ),
            )

        conn.commit()

        total_attempts = len(system_discovered)
        successful_attempts = [
            snapshot for snapshot in system_discovered if snapshot.get("is_better_deal")
        ]
        failed_attempts = [
            snapshot
            for snapshot in system_discovered
            if not snapshot.get("is_better_deal")
        ]
        # return_message = f"{total_attempts} deals saved. {len(successful_attempts)}/{total_attempts} better deals, {len(failed_attempts)}/{total_attempts} worse deals."
        # print(return_message)
        return (
            session_id,
            total_attempts,
            len(successful_attempts),
            len(failed_attempts),
        )
    except Exception as e:
        print(f"save_discovery_session: DB error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fd_discovery_entry_node(
    state: State, config: RunnableConfig
) -> Command[Literal[FD_BASELINE_ENTRY_NODE, FD_DISCOVERY_AGENT]]:
    # print(
    #     "-------------------------------- HITTTTTTT fd_discovery_entry_node --------------------------------"
    # )
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
            # print("initializing system_discovered list with a new snapshot")

            # Clear prior discovery chat but keep the latest ToolMessage if present
            human_message = HumanMessage(
                content="Control tranferred from baseline agent to discovery agent. Start flight discovery process.",
            )

            # For DEMO ONLY !!!!!!!!!!!!!!!!!!!!!!!!
            baseline.get("flights")[0].min_amount += 30
            # For DEMO ONLY !!!!!!!!!!!!!!!!!!!!!!!!

            updates = {"fd_session_info": session_info}
            updates["flight_discovery_messages"] = ["__RESET__", human_message]

            return Command(update=updates)
            # return Command(goto=END,update=updates)

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
                # print(
                #     "current snapshot has 3/3 fields(flights, filter_used, comment), 2 steps done. Append a new snapshot, move on to next round of discovery."
                # )

                review_result = review_discovery_session(state)
                if review_result == END_DISCOVERY:
                    # print("Review result: End discovery")
                    sess_id, total_attempts, successful_attempts, failed_attempts = (
                        save_discovery_session(state)
                    )
                    return_message = (
                        "Flight discovery session (session_id="
                        + sess_id
                        + ") Completed with "
                        + str(total_attempts)
                        + " attempts, "
                        + str(successful_attempts)
                        + " successful attempts, "
                        + str(failed_attempts)
                        + " failed attempts."
                    )
                    # print(return_message)
                    return Command(
                        goto=FD_DISCOVERY_AGENT,
                        update={
                            "flight_discovery_messages": [
                                HumanMessage(
                                    content="Search session completed. Call 'WorkerCompleteOrEscalate' with reason: "
                                    + return_message,
                                )
                            ]
                        },
                    )
                else:
                    # print("Review result: Continue discovery")
                    return Command(
                        goto=FD_DISCOVERY_AGENT,
                        update={"fd_session_info": session_info},
                    )

            if (
                cur_snapshot.get("flights") is not None
                and cur_snapshot.get("filter_used") is not None
                and cur_snapshot.get("comment") is None
            ):
                # print(
                #     "current snapshot has 2/3 fields(flights, filter_used), missing comment. Discovery step done. Move on to evaluation step."
                # )
                return Command(goto=FD_DISCOVERY_AGENT)
            else:
                # print("Fall back. un-handled current snapshot values.")
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
