from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import uuid
from primary_assistant_tools import WorkerCompleteOrEscalate
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from typing import Literal, Optional, Annotated
from amadeus_api import (
    get_amadeus_client,
    normalize_offers_for_tool,
    rank_flights,
)
from primary_assistant_chain import FDSnapshot, State, FDSessionInfo
from langchain_core.runnables import RunnableConfig
from graph_setup import FD_DISCOVERY_ENTRY_NODE, FD_RETURN_NODE


llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")


def _coerce_dt(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        v = value.replace("Z", "")
        try:
            return datetime.fromisoformat(v)
        except Exception:
            pass
    raise ValueError(f"Invalid datetime value: {value!r}")


@tool("search_flight")
def search_flight(
    departure_airport: str,
    arrival_airport: str,
    departure_time: list[datetime],
    budget: Optional[int] = None,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Search best flight by provided filter criteria, result will be appended to flight candidates.
    Parameters:
    departure_airport : str
        IATA/ICAO code, e.g. 'YUL', 'YHU'. Must match exactly. only 1 airport code is allowed at a time.
    arrival_airport : str
        IATA/ICAO code, e.g. 'YYZ'. Must match exactly. only 1 airport code is allowed at a time.
    departure_time : list[datetime]
        A two-item list [window_start, window_end]. Each item can be a `datetime` or an ISO 8601 string.
    budget : Optional[int]
        Optional maximum ticket price. If provided, we filter normalized offers by their `min_amount`.
    """

    if not isinstance(departure_time, (list, tuple)) or len(departure_time) != 2:
        return {
            "error": "departure_time must be a two-item list: [window_start, window_end]."
        }

    try:
        window_start = _coerce_dt(departure_time[0])
        window_end = _coerce_dt(departure_time[1])
    except Exception as e:
        return {"error": f"Invalid departure_time window: {e}"}
    # Ensure start <= end (swap if user provided reversed order)
    if window_start > window_end:
        window_start, window_end = window_end, window_start

    client = get_amadeus_client()
    offers = client.amadeus_search_flights(
        origin=departure_airport,
        destination=arrival_airport,
        departure_time_window=[window_start, window_end],
        adults=1,
        max_results_per_day=50,
    )

    # Check if offers is an error response
    if isinstance(offers, dict) and offers.get("error"):
        return f"Flight search failed: {offers.get('message', 'Unknown error')}"

    # Ensure offers is a list before normalizing
    if not isinstance(offers, list):
        return f"Unexpected response format from flight search: {type(offers)}"

    normalized = normalize_offers_for_tool(offers)
    if budget is not None:
        try:
            budget = int(budget)

            def _min_amt(x):
                try:
                    return getattr(x, "min_amount")
                except Exception:
                    return (x or {}).get("min_amount")

            normalized = [
                offer
                for offer in normalized
                if _min_amt(offer) is not None and _min_amt(offer) <= budget
            ]
        except Exception:
            pass

    filters_used = {
        "departure_airport": departure_airport,
        "arrival_airport": arrival_airport,
        "departure_time_window": [
            window_start.isoformat(sep=" "),
            window_end.isoformat(sep=" "),
        ],
        "budget": budget,
    }

    flight_ranked = rank_flights(filters_used, normalized, 5)
    flight = flight_ranked[0]["flight"] if flight_ranked else None

    if flight is None:
        print(f"🔧 search_flight tool returning Command with 'No flight found'")
        return {
            "flight_discovery_messages": [
                ToolMessage(content="No flight found", tool_call_id=tool_call_id)
            ]
        }
        # return Command(
        #     update={
        #         "flight_discovery_messages": [
        #             ToolMessage(content="No flight found", tool_call_id=tool_call_id)
        #         ]
        #     }
        # )

    snapshot_id = str(uuid.uuid4())
    candidate = FDSnapshot(
        id=snapshot_id,
        filter_used=filters_used,
        flights=[flight] if flight else [],
    )
    # Example of using a ternary operator (conditional expression) in Python:
    # result = value_if_true if condition else value_if_false
    #
    # For example, to set a variable 'status' based on whether 'flight' is not None:
    tool_message = (
        ToolMessage(content=f"Flight found: {snapshot_id}", tool_call_id=tool_call_id)
        if flight
        else ToolMessage(content="No flight found", tool_call_id=tool_call_id)
    )

    print(f"🔧 search_flight tool returning Command with candidate ID: {snapshot_id}")
    return {
        "baseline_flight_candidates": [candidate],
        "flight_discovery_messages": [tool_message],
    }
    return Command(
        update={
            "baseline_flight_candidates": [candidate],
            "flight_discovery_messages": [tool_message],
        }
    )


# result = search_flight.invoke(
#     {
#         "departure_airport": "YHU",
#         "arrival_airport": "YTZ",
#         "departure_time": [
#             datetime(2025, 8, 15, 12, 0, 0),
#             datetime(2025, 8, 15, 18, 0, 0),
#         ],
#         "budget": 1000,
#         "tool_call_id": "manual-run-1",  # required because your tool expects InjectedToolCallId
#         "state": {},
#         "config": {},
#     }
# )
# print(result)


@tool("evaluate_flight")
def evaluate_flight(
    snapshot_id: str,
    comment: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
):
    """
    Evaluate a flight by providing a comment. this comment should reflect yout thoughts on the flights found, and how to make update the filter to make next flight cheaper.

    Parameters:
    snapshot_id: str
        The id of the snapshot to update.
    comment: str
        The comment to store in `comment`.
    """
    # Pull the current session info safely
    session = state.get("fd_session_info") or {}
    baseline = session.get("baseline") or {}
    system_discovered = session.get("system_discovered") or []

    target_id = str(snapshot_id)
    updated = False

    # Update baseline if it matches
    try:
        if baseline and str(baseline.get("id")) == target_id:
            baseline["comment"] = comment
            updated = True
    except AttributeError:
        # If baseline isn't a dict-like structure, ignore gracefully
        baseline = baseline or {}

    # Otherwise update a matching snapshot in system_discovered
    if not updated and isinstance(system_discovered, list):
        for snapshot in system_discovered:
            try:
                if str(snapshot.get("id")) == target_id:
                    snapshot["comment"] = comment
                    updated = True
                    break
            except AttributeError:
                # Skip non-dict entries gracefully
                continue

    if updated:
        # Only include keys that have values to respect the TypedDict's optional fields
        new_info = {}
        if baseline:
            new_info["baseline"] = baseline
        if system_discovered:
            new_info["system_discovered"] = system_discovered

        print(
            f"🔧 evaluate_flight tool returning Command with updated comment for ID: {target_id}"
        )
        return {
            "fd_session_info": new_info,
            "flight_discovery_messages": [
                ToolMessage(
                    content="Updated comment for snapshot: " + target_id,
                    tool_call_id=tool_call_id,
                )
            ],
        }
        return Command(
            update={
                "fd_session_info": new_info,
                "flight_discovery_messages": [
                    ToolMessage(
                        content="Updated comment for snapshot: " + target_id,
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    else:
        return {
            "flight_discovery_messages": [
                ToolMessage(
                    content="No snapshot found in fd_session_info with id: "
                    + target_id,
                    tool_call_id=tool_call_id,
                )
            ],
        }
        return Command(
            update={
                "flight_discovery_messages": [
                    ToolMessage(
                        content="No snapshot found in fd_session_info with id: "
                        + target_id,
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def finalize_baseline(
    best_candidate_id: str,
    comment: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> Command[Literal[FD_DISCOVERY_ENTRY_NODE, FD_RETURN_NODE]]:
    """
    End the baseline flight discovery process by providing the best flight's id from flight candidates.
    Generate a short and concise comment to the subsequent assistant who take your advice and apply new filter to find cheaper flights.
    Parameters:
    best_candidate_id: str
        The id of the best flight candidate you choose.
    comment: str
        comment should 1) give context: reflect your thoughts on the baseline filter used and baseline flight chosen, 2) give direction: suggestion on how to improve the filter to find cheaper flights(suggestion must be made based on availabel filter criteria in FlightRequirements).
    """
    candidates = state.get("baseline_flight_candidates", [])
    matched = [c for c in candidates if str(c.get("id")) == str(best_candidate_id)]
    baseline_snapshot = matched[0] if matched else None

    if baseline_snapshot is None:
        return {
            "flight_discovery_messages": [
                ToolMessage(
                    content="No candidate flight found for provided best_candidate_id: "
                    + str(best_candidate_id),
                    tool_call_id=tool_call_id,
                )
            ],
        }
        return Command(
            goto=FD_RETURN_NODE,
            update={
                "flight_discovery_messages": [
                    ToolMessage(
                        content=(
                            "No candidate flight found for provided best_candidate_id: "
                            + str(best_candidate_id)
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            },
        )

    baseline_snapshot["comment"] = comment

    new_fd_session_info = FDSessionInfo(baseline=baseline_snapshot)

    # Save the completed baseline state to MongoDB for future debugging
    try:
        # Create the updated state that would be returned
        updated_state = {
            **state,  # Current state
            "fd_session_info": new_fd_session_info,
            "flight_discovery_messages": state.get("flight_discovery_messages", [])
            + [
                ToolMessage(
                    content="Flight discovery process finished with best flight: "
                    + str(baseline_snapshot.get("id")),
                    tool_call_id=tool_call_id,
                )
            ],
        }

        # saved_state_id = save_baseline_completed_state(
        #     updated_state,
        #     config,
        #     f"Baseline completed - Flight {baseline_snapshot.get('id')} selected",
        # )
        # print(f"!!!!!!!!! 💾 State saved with ID: {saved_state_id}")

    except Exception as e:
        print(f"⚠️  Failed to save baseline state: {e}")
        # Continue execution even if saving fails

    print(f"🔧 finalize_baseline tool returning Command with updated fd_session_info")
    return Command(
        update={
            "fd_session_info": new_fd_session_info,
            "flight_discovery_messages": [
                ToolMessage(
                    content="Flight discovery process finished with best flight: "
                    + str(baseline_snapshot.get("id")),
                    tool_call_id=tool_call_id,
                )
            ],
        },
        goto=FD_DISCOVERY_ENTRY_NODE,
    )


fd_baseline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a baselineflight discovery agent. Your job is: to take user's FlightRequirements as filter criteria, trying out different combinations of filter criteria to find the best flight for user.\n"
            # "You are a flight discovery agent. Your job is: come up with system-recommended search criteria base on user's basic search criteria, and find flights with lower price for user.\n"
            "CONTEXT:\n"
            "Flight Search Requirement: is a filter user specifies to search for flights. Each search critria has a priority level(priority_1, priority_2), this priority level is used to determine a requirements value's importance/preference to user.\n"
            "Requirement with smaller number(e.g. priority_1) is user's most preferred/highest need. requirement with larger number(e.g. priority_4) is user's least preferred/lowest need.\n"
            "Flight Requirements:\n{flight_requirements}\n"
            "Current flight candidates: {baseline_flight_candidates}\n"
            "Current time: {time}\n\n"
            "Flight Discovery Process:\n"
            "step 1: search flight with base filter: the filter applies only priority_1 criteria/requirements.\n"
            "if base filter returns a result(shown in flight candidates), it's the best flight, start the finalizing baseline process.\n"
            "step 2: if no results returned from step 1, search flights with new filters by only modifying 1 criteria to new priority level at a time."
            "for example, change departure time from priority_1 to priority_2, and keep rest of filter same as base filter. next time, change departure time from priority_2 to priority_3. In this process, all filter criterias must remain in priority_1, change only 1 criteria at a time.\n"
            "repeat this process until you have used every criteria and priority level in FlightRequirements once.\n"
            "if flights found in flight candidates, start the finalizing baseline process.\n"
            "step 3: if still no flights returned from step 2. you can apply any combination of filter criteria to find flight. you are not restricted to only changing 1 filter criteria like step 2, you can change multiple filter criteria at a time.\n"
            "step 4: make 3 more attempts to find flights, if still no flights returned, call WorkerCompleteOrEscalate and abort the task.\n"
            "if you find flights from candidates, start the finalizing baseline process\n"
            "Finalizing Baseline Process:\n"
            "among all the 'flight candidates' you see, use your common sense to selcet a best flight for user as baseline. call finalize_baseline, and provide a comment.\n"
            "after 'finalize_baseline' called, job will be handed off to a subsequent assistant who will take baseline filter and flihgt as reference, apply filter criteria NOT in baseline filter to find better flights.\n"
            "Rules:\n"
            "- You must only search flights using the filter criteria in FlightRequirements, do not make up any filter criteria.\n"
            "- You do not need to present the flight candidates to user, just keep searching flights until you have found the best flight.\n"
            "- the comment you provide in 'finalize_baseline' should be short and concise, should not give technical details like 'priority_1', 'base filter' or exact filter values, you should only give a high level overview and guidance, for example: 'consider a much later departure time, current departure time is too early'\n",
        ),
        ("placeholder", "{flight_discovery_messages}"),
    ]
).partial(time=datetime.now().isoformat())


@tool("WorkerCompleteOrEscalate")
def worker_complete_or_escalate(
    cancel: bool,
    reason: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    A tool to mark the current task as completed and/or to escalate control of the dialog to the primary assistant,
    who can re-route the dialog based on the user's needs.
    You must call this tool when you have completed the current task.
    example 1: "cancel": True, "reason": "User changed their mind about the current task.",
    example 2: "cancel": False, "reason": "I have fully completed the task: <task_description>",
    example 3: "cancel": True, "reason": "I need help to search the user's emails or calendar for more information.",
    """
    print(
        f"🔧 worker_complete_or_escalate tool called with cancel: {cancel}, reason: {reason}"
    )
    return


baseline_tools = [
    search_flight,
    worker_complete_or_escalate,
    finalize_baseline,
    worker_complete_or_escalate,
]
flight_discovery_chain = fd_baseline_prompt | llm.bind_tools(baseline_tools)
