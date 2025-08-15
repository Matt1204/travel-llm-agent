from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import uuid
from primary_assistant_tools import WorkerCompleteOrEscalate
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from typing import Optional, Dict, Any, Annotated
from fd_baseline_chain import search_flight, evaluate_flight, _coerce_dt
from primary_assistant_chain import FDSnapshot, State, FDSessionInfo
from langchain_core.runnables import RunnableConfig
import json
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from amadeus_api import (
    get_amadeus_client,
    normalize_offers_for_tool,
    rank_flights,
)

llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")


# ---- selectors from State
def _to_jsonable(value):
    """Recursively convert objects (e.g., Pydantic models, datetimes) to JSON-serializable types."""
    try:
        # Pydantic v2 models
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
    except Exception:
        pass

    # Datetime handling
    if isinstance(value, datetime):
        try:
            return value.isoformat(sep=" ")
        except Exception:
            return str(value)

    # Containers
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [
            _to_jsonable(v)
            for v in (list(value) if not isinstance(value, list) else value)
        ]

    # Primitives or unknowns
    return value


def extract_baseline_flight_snapshot(state: State) -> str:
    session_info = state.get("fd_session_info") or {}
    baseline = session_info.get("baseline") or {}
    if baseline:
        safe_baseline = _to_jsonable(baseline)
        return json.dumps(safe_baseline, indent=2)
    else:
        return "Error. no baseline snapshot found"


def extract_current_flight_snapshot(state: State) -> str:
    session_info = state.get("fd_session_info") or {}
    system_discovered = session_info.get("system_discovered") or []
    if len(system_discovered) > 0:
        safe_snapshot = _to_jsonable(system_discovered[-1])
        return json.dumps(safe_snapshot, indent=2)
    else:
        return "Error. No current_flight_snapshot found, empty system_discovered list"


fd_prompt_inputs = {
    # pass through the running message thread for this assistant
    "flight_discovery_messages": itemgetter("flight_discovery_messages"),
    # computed fields (to strings) from selectors above
    "baseline_flight_snapshot": RunnableLambda(extract_baseline_flight_snapshot),
    "current_flight_snapshot": RunnableLambda(extract_current_flight_snapshot),
}


fd_discovery_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a flight Recommendation agent. Your job is: to take user's baseline filter, coming up with new system-recommended filter criteria(values) to find cheaper flights for user.\n"
            "You will repeatedly do step 1) propose new filter and search fligths ('update_filter_and_search_flight').  2) evaluate/review your previous serach process('evaluate_flight'). You will not notify user your progress, keep working until you are told to stop\n"
            "CONTEXT:\n"
            # "baseline_filter: is the filter used to find the so far the best flight for user, you will work base on base filter, introduce new filter values to this baseline filter to find a cheaper flight.\n"
            # "baseline_flight: is the flight found by baseline filter, your goal is to find a flight cheaper than baseline flight.\n"
            # "system_discovered: is the list of flights you have found so far.\n"
            # "filter_in_use: is the filter you are using to find a cheaper flight, you will update this filter to find a cheaper flight.\n"
            # "flights_found: is the list of flights found using the filter_in_use.\n"
            "- snapshot: contains the information of a flight search attempt, including filter_used, flights(flights returned by filter_used), comment, is_better_deal(whether the flights found are cheaper than baseline flight)\n"
            "- 'baseline_flight_snapshot' is the benchmark snapshot of so far the best fit flight for user, you will keep working on basis of this baseline filter, introduce new filter values to find a cheaper flight than baseline flight\n"
            "'baseline_flight_snapshot' is the best-fit for user, but it may not be the cheapest flight for user. So you only have 1 goal: find a cheaper flight than baseline flight\n"
            "- 'current_flight_snapshot' is the snapshot of the current flight search attempt, which relfect your latest changes to the filter and flights found\n"
            "Current time: {time}\n\n"
            "<baseline_flight_snapshot>{baseline_flight_snapshot}</baseline_flight_snapshot>\n"
            # "system_discovered: {system_discovered}\n\n"
            "<current_flight_snapshot>{current_flight_snapshot}</current_flight_snapshot>\n"
            "Flight Recommendation Process:\n"
            "step 1: recommend new filter values and search for flights by calling 'update_filter_and_search_flight' with your proposed filter values, after this, <current_flight_snapshot> will be updated with the filter you proposed(filter_used), and flights(if any) found.\n"
            "To propose a new filter: Use your common sense + analysis of comments from previous search snapshot, propose a new filter that is likely to find a cheaper flight than baseline flight.\n"
            "For example, if the baseline filter's departure_time is today, then you may suggest changing the departure_time to a day in the future, since flights departing further ahead may be priced lower than near-term departures.\n"
            "For example, if the baseline filter's departure_airport is a small airport, then you may suggest changing the departure_airport to a large hub airport nearby, since flights departing from larger airports may be priced lower than small airports.\n"
            "For example, if you discovered from orevious searches that you are not getting any flights by updating the departure_time, then you may suggest changing the airport\n"
            "Step 2: evaluate the new flights found and filter used; with filter proposed and flights searched, review them. Call 'evaluate_flight', and provide a thoughts on the <current_flight_snapshot>, this comment will be used as a guidance for future search attempt.\n"
            "Repeating step 1 and step 2, until you are told to stop.\n"
            "Rules:\n"
            "- You should suggest new filter value(s) that is likely to find a cheaper flight than baseline flight. "
            "- When proposing new filter, start by making small changes to the baseline filter, and then gradually increase the changes to find a cheaper flight. For example, start by only changing 1 filter value, and then gradually increase the changes to find a cheaper flight.\n"
            "- When reviewing the current_flight_snapshot, you should 1) generate yout reflection and insights on the <current_flight_snapshot>, 2) give advice on how to improve the filter to find a cheaper flight, do not propose filter value directly, just give general guidanve on the direction to improve the filter. \n"
            "- You do not need to present the flight candidates to user, just keep searching flights until you have found the best flight.\n",
        ),
        ("placeholder", "{flight_discovery_messages}"),
    ]
).partial(time=datetime.now().isoformat())


@tool
def update_filter_and_search_flight(
    departure_airport: str,
    arrival_airport: str,
    departure_time: list[datetime],
    budget: Optional[int] = None,
    why_this_filter: str = "",
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
):
    """
    Step 1: update the filter_used in current_flight_snapshot, and search for flights with the new filter criteria.
    Parameters:
    departure_airport : str
        IATA/ICAO code, e.g. 'YUL', 'YHU'. Must match exactly. only 1 airport code is allowed at a time.
    arrival_airport : str
        IATA/ICAO code, e.g. 'YYZ'. Must match exactly. only 1 airport code is allowed at a time.
    departure_time : list[datetime]
        A two-item list [window_start, window_end]. Each item can be a `datetime` or an ISO 8601 string.
    reason: str
        A short and concise reason on why you are proposing this filter.
    """
    print(
        f"----update_filter_and_search_flight: {departure_airport} {arrival_airport} {departure_time} {budget}----"
    )
    print(f" -- {why_this_filter} --")
    session_info = state.get("fd_session_info") or {}
    system_discovered = session_info.get("system_discovered") or []
    cur_snapshot = system_discovered[-1]

    try:
        window_start = _coerce_dt(departure_time[0])
        window_end = _coerce_dt(departure_time[1])
    except Exception as e:
        return {"error": f"Invalid departure_time window: {e}"}
    # Ensure start <= end (swap if user provided reversed order)
    if window_start > window_end:
        window_start, window_end = window_end, window_start

    new_filter = {
        "departure_airport": departure_airport,
        "arrival_airport": arrival_airport,
        "departure_time_window": [
            window_start.isoformat(sep=" "),
            window_end.isoformat(sep=" "),
        ],
        "budget": budget,
    }
    cur_snapshot["filter_used"] = new_filter

    client = get_amadeus_client()
    offers = client.amadeus_search_flights(
        origin=departure_airport,
        destination=arrival_airport,
        departure_time_window=[window_start, window_end],
        adults=1,
        max_results_per_day=50,
    )

    normalized = normalize_offers_for_tool(offers)

    if budget is not None:
        try:
            budget = int(budget)
            normalized = [
                offer
                for offer in normalized
                if getattr(offer, "min_amount", None) is not None
                and getattr(offer, "min_amount", None) <= budget
            ]
        except Exception:
            pass
    flight_ranked = rank_flights(new_filter, normalized, 5)
    flights_found = [f["flight"] for f in flight_ranked]
    num_flights_found = len(flights_found)

    baseline_snapshot = session_info.get("baseline") or {}
    baseline_flight = baseline_snapshot.get("flights")[0]
    baseline_price = getattr(baseline_flight, "min_amount", None)
    better_flights = [f for f in flights_found if f.get("min_amount") < baseline_price]
    if len(better_flights) > 0:
        cur_snapshot["flights"] = better_flights
        cur_snapshot["is_better_deal"] = True
        tool_message = ToolMessage(
            content=f"{len(better_flights)} flights found that are better than baseline flight. using filter: {new_filter}",
            tool_call_id=tool_call_id,
        )
        return Command(
            update={
                "fd_session_info": session_info,
                "flight_discovery_messages": [tool_message],
            }
        )
    else:
        cur_snapshot["flights"] = []
        cur_snapshot["is_better_deal"] = False
        tool_message = ToolMessage(
            content=f"{num_flights_found} flights found, but {len(better_flights)} are cheaper than baseline flight. using filter: {new_filter}",
            tool_call_id=tool_call_id,
        )
        return Command(
            update={
                "fd_session_info": session_info,
                "flight_discovery_messages": [tool_message],
            }
        )


# update_filter_and_search_flight.invoke(
#     {
#         "departure_airport": "YUL",
#         "arrival_airport": "YTZ",
#         "departure_time": [
#             datetime(2025, 8, 17, 0, 0, 0),
#             datetime(2025, 8, 17, 23, 59, 59),
#         ],
#         "budget": 1000,
#         "why_this_filter": "debugging.....",
#         "tool_call_id": "manual-run-1",  # required because your tool expects InjectedToolCallId
#         "state": {},
#         "config": {},
#     }
# )


@tool
def evaluate_flight(
    snapshot_id: str,
    comment: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
):
    """
    Step 2: Evaluate the current_flight_snapshot by providing its id, a short and concise comment.
    This comment should reflect your thoughts on the flights found(success or failure), and general direction on how to improve the filter to find a cheaper flight.
    Parameters:
    snapshot_id: str
        The id of the current_flight_snapshot.
    comment: str
        A comment on the current_flight_snapshot.
    """
    print(f"----evaluate_flight: {snapshot_id}----")
    session_info = state.get("fd_session_info") or {}
    system_discovered = session_info.get("system_discovered") or []
    cur_snapshot = system_discovered[-1]

    if (
        cur_snapshot.get("flights") is not None
        and cur_snapshot.get("filter_used") is not None
    ):
        cur_snapshot["comment"] = comment
        return Command(
            update={
                "fd_session_info": session_info,
                "flight_discovery_messages": [
                    ToolMessage(
                        content=f"Comment updated for snapshot: {cur_snapshot}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    else:
        print("Error: there is no flights and filter_used in current_flight_snapshot")
        return Command(
            update={
                "flight_discovery_messages": [
                    ToolMessage(
                        content=f"Error:No snapshot has flights or/and filter_used equals None. re-run update_filter_and_search_flight to find a cheaper flight.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


fd_discovery_tools = [
    update_filter_and_search_flight,
    evaluate_flight,
    WorkerCompleteOrEscalate,
]
fd_discovery_chain = (
    fd_prompt_inputs | fd_discovery_prompt | llm.bind_tools(fd_discovery_tools)
)
