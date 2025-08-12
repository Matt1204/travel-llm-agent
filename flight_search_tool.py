from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import json
from amadeus_api import (
    get_amadeus_client,
    normalize_offers_for_tool,
    rank_flights,
)
from langchain_core.tools import tool, InjectedToolCallId
import sqlite3
from langgraph.graph import END
from langgraph.types import Command
from db_connection import db_file
from langgraph.prebuilt import InjectedState
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
import uuid
from langchain_core.messages import ToolMessage


@tool
def search_flights(
    departure_airport: str,
    arrival_airport: str,
    departure_time: list[datetime],
    budget: Optional[int] = None,
    top_k: Optional[int] = 3,
):
    """
    Search flights by exact departure/arrival airport match and a departure-time window, using
    Amadeus Flight Offers Search under the hood (aggregated per day over the window).

    Parameters:
    departure_airport : str
        IATA/ICAO code, e.g. 'YUL', 'YHU'. Must match exactly.
    arrival_airport : str
        IATA/ICAO code, e.g. 'YYZ'. Must match exactly.
    departure_time : list[datetime]
        A two-item list [window_start, window_end]. Each item can be a `datetime` or an ISO 8601 string.
        Only offers with first itinerary departure within this window (inclusive) are returned.
    budget : Optional[int]
        Optional maximum ticket price. If provided, we filter normalized offers by their `min_amount`.
    top_k : Optional[int]
        If provided, compute grouped rankings: top K direct flights (0 connections) and top K flights with
        transfer (>=1 connection), using price, duration, and stops. Results returned under `ranked_results`
        as { "direct": [...], "transfer": [...] } with scores and metrics.

    Returns:
    dict
        A JSON-serializable dict with the applied filters and a list of matching offers in a format
        similar to the previous DB-backed tool (keys like flight_no, scheduled_departure, fares, min_amount).
        Also includes the raw Amadeus payload under `raw` for downstream usage.
    """
    # --- validate & coerce the time window ---
    if not isinstance(departure_time, (list, tuple)) or len(departure_time) != 2:
        return {
            "error": "departure_time must be a two-item list: [window_start, window_end]."
        }

    def _coerce_dt(value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # Be lenient with trailing 'Z'
            v = value.replace("Z", "")
            try:
                return datetime.fromisoformat(v)
            except Exception:
                pass
        raise ValueError(f"Invalid datetime value: {value!r}")

    try:
        window_start = _coerce_dt(departure_time[0])
        window_end = _coerce_dt(departure_time[1])
    except Exception as e:
        return {"error": f"Invalid departure_time window: {e}"}

    # Ensure start <= end (swap if user provided reversed order)
    if window_start > window_end:
        window_start, window_end = window_end, window_start

    # --- call Amadeus over the window ---
    client = get_amadeus_client()
    offers = client.amadeus_search_flights(
        origin=departure_airport,
        destination=arrival_airport,
        departure_time_window=[window_start, window_end],
        adults=1,
        currency="CAD",
        non_stop=None,
        included_airline_code=None,
        max_results_per_day=50,
        additional_params=None,
    )

    # offers = (res or {}).get("data") or []
    normalized = normalize_offers_for_tool(offers)

    # Optional budget filtering on normalized offers
    if budget is not None:
        try:
            b = int(budget)
            normalized = [o for o in normalized if o.get("min_amount") <= b]
        except Exception:
            # If budget is malformed, ignore filtering rather than fail hard
            pass

    # Sort by departure time then price if available
    # def _to_dt(val):
    #     if not val:
    #         return datetime.max
    #     s = str(val).replace("Z", "")
    #     try:
    #         return datetime.fromisoformat(s)
    #     except Exception:
    #         return datetime.max

    # normalized.sort(
    #     key=lambda i: (
    #         _to_dt(i.get("scheduled_departure")),
    #         i.get("min_amount") or 10**12,
    #     )
    # )

    filters_applied = {
        "departure_airport": departure_airport,
        "arrival_airport": arrival_airport,
        "departure_time_window": [
            window_start.isoformat(sep=" "),
            window_end.isoformat(sep=" "),
        ],
        "budget": budget,
    }

    response: Dict[str, Any] = {
        "filters_applied": filters_applied,
    }

    if isinstance(top_k, int) and top_k > 0:
        try:

            # def _total_connections(f: Dict[str, Any]) -> int:
            #     offer = f.get("raw") or {}
            #     itineraries = offer.get("itineraries") or []
            #     total = 0
            #     for it in itineraries:
            #         segs = it.get("segments") or []
            #         total += max(0, len(segs) - 1)
            #     return total

            direct_flights = [f for f in normalized if f["is_direct"]]
            transfer_flights = [f for f in normalized if not f["is_direct"]]

            ranked_results: Dict[str, Any] = {}
            if direct_flights:
                ranked_direct = rank_flights(filters_applied, direct_flights, top_k)
                if ranked_direct:
                    ranked_results["direct"] = ranked_direct
            if transfer_flights:
                ranked_transfer = rank_flights(filters_applied, transfer_flights, top_k)
                if ranked_transfer:
                    ranked_results["transfer"] = ranked_transfer

            if ranked_results:
                response["ranked_results"] = ranked_results
        except Exception:
            # If ranking fails for any reason, omit ranked_results gracefully
            pass

    return response


search_flights.invoke(
    {
        "departure_airport": "LHR",
        "arrival_airport": "CDG",
        "departure_time": [
            datetime(2025, 8, 19, 0, 0, 0),
            datetime(2025, 8, 20, 0, 0, 0),
        ],
        "budget": 1000,
        "top_k": 10,
    }
)


@tool
def get_flight_details(
    departure_airport: str,
    arrival_airport: str,
    departure_time: list[datetime],
    included_airline_code: Optional[str] = None,
    non_stop: Optional[bool] = False,
    flight_number: Optional[str] = None,
):
    """
    Return detailed information for a *specific* flight offer, you should provide the parameters that most cloesly describe the flight user is looking for, not the flight requirement.
    parameters: must be provided to best describe the flight user is looking for, not the flight requirement.
    - departure_airport: IATA code (e.g., "YUL")
    - arrival_airport: IATA code (e.g., "YYZ")
    - departure_time: [window_start, window_end] as datetimes or ISO strings
    - included_airline_code: two-letter IATA airline code to restrict results, first 2 letters of flight number (e.g., "DY")
    - non_stop: if True, only non-stop flights are considered. select the value of the flight you are looking for.
    - flight_number: flight number to restrict results (e.g., "DY1126")

    Returns a dict with the applied filters, the selected offer, and the priced payload.
    """
    # call get_flight_details tool with parameters: departure_airport=OSL, arrival_airport=HAM, departure_time=2025-08-14, included_airline_code=DY, non_stop=True
    # --- validate & coerce the time window ---
    if not isinstance(departure_time, (list, tuple)) or len(departure_time) != 2:
        return {
            "error": "departure_time must be a two-item list: [window_start, window_end]."
        }

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

    try:
        window_start = _coerce_dt(departure_time[0])
        window_end = _coerce_dt(departure_time[1])
    except Exception as e:
        return {"error": f"Invalid departure_time window: {e}"}

    if window_start > window_end:
        window_start, window_end = window_end, window_start

    client = get_amadeus_client()

    # --- iterate days in the window and collect offers ---
    matching_offers: List[Dict[str, Any]] = []

    def _first_departure_time(offer: Dict[str, Any]):
        try:
            itins = offer.get("itineraries") or []
            segs0 = itins[0].get("segments") if itins else None
            dep_at = (segs0[0].get("departure") or {}).get("at") if segs0 else None
            if not dep_at:
                return None
            return datetime.fromisoformat(str(dep_at).replace("Z", ""))
        except Exception:
            return None

    d = window_start.date()
    end_date = window_end.date()
    while d <= end_date:
        res = client.search_flight_offers_by_day(
            origin=departure_airport,
            destination=arrival_airport,
            departure_date=d,
            adults=1,
            currency="CAD",
            non_stop=non_stop,
            max_results=5,
            included_airline_code=included_airline_code,
            additional_params=None,
        )
        if isinstance(res, dict) and res.get("error"):
            # Bubble up the first Amadeus error
            return res

        data = (res or {}).get("data") or []
        for offer in data:
            dep_dt = _first_departure_time(offer)
            if dep_dt is not None:
                if window_start <= dep_dt <= window_end:
                    matching_offers.append(offer)
            else:
                # If we can't parse the timestamp, keep it as a candidate (best-effort)
                matching_offers.append(offer)

        d = d + timedelta(days=1)

    # --- optional filter by marketing flight number (carrierCode + number) ---
    # Per Amadeus FlightSegment fields in the API docs:
    #   - carrierCode: two-letter airline code (e.g., "DY")
    #   - number: flight number string (e.g., "1126")
    # We normalize as UPPERCASE and remove non-alphanumeric characters, then compare equality
    # against the provided `flight_number` parameter. If any segment in an offer matches,
    # we keep that offer.
    if flight_number:

        def _normalize_fno(s: str) -> str:
            return "".join(ch for ch in str(s).upper() if ch.isalnum())

        target_flight_no = _normalize_fno(flight_number)

        def find_flight_no_from_offer(offer: Dict[str, Any]) -> List[str]:
            nums: List[str] = []
            for itin in offer.get("itineraries") or []:
                for seg in itin.get("segments") or []:
                    cc = (seg.get("carrierCode") or "").strip().upper()
                    num = str(seg.get("number") or "").strip()
                    if cc and num:
                        nums.append(f"{cc}{num}")
            return nums

        # filtered_offers: List[Dict[str, Any]] = []
        for offer in matching_offers:
            flight_no_offer = find_flight_no_from_offer(offer)
            normalized_nums = {_normalize_fno(n) for n in flight_no_offer}

            if target_flight_no in normalized_nums:
                matching_offers = [offer]
                break

    if not matching_offers:
        return {
            "filters_applied": {
                "departure_airport": departure_airport,
                "arrival_airport": arrival_airport,
                "departure_time_window": [
                    window_start.isoformat(sep=" "),
                    window_end.isoformat(sep=" "),
                ],
                "included_airline_code": included_airline_code,
                "non_stop": non_stop,
            },
            "error": "No matching offers found for the provided filters/time window.",
        }

    matched_offer = matching_offers[0]

    # --- Step 3: confirm detailed pricing for the selected offer ---
    priced = client.price_flight_offers([matched_offer])

    response: Dict[str, Any] = {}

    if isinstance(priced, dict) and priced.get("error"):
        response["pricing_error"] = priced
        return response

    priced = {
        key: val
        for (key, val) in priced["data"].items()
        if key != "bookingRequirements"
    }
    response["priced"] = priced
    response["flight_number"] = flight_number
    return response


@tool
def get_flight_info_by_id(flight_id: int):
    """
    Retrieve detailed information for a single flight by `flight_id`.

    Parameters:
    flight_id : int
        The flight_id of the flight to retrieve information for.

    Returns:
    dict
        JSON-serializable object with a `flight_id` key entry containing the flight metadata
            and a `fares` list (each with `fare_conditions` and `amount`). If not found,
        returns {"error": "..."}.
    """
    try:
        flight_id_int = int(flight_id)
    except Exception:
        return {"error": f"flight_id must be an integer, got: {flight_id!r}"}

    # --- SQL: fetch flight row + all fares ---
    sql = """
        SELECT
            f.flight_id,
            f.flight_no,
            f.scheduled_departure,
            f.scheduled_arrival,
            f.departure_airport,
            f.arrival_airport,
            f.status,
            f.aircraft_code,
            f.actual_departure,
            f.actual_arrival
        FROM flights AS f
        WHERE f.flight_id = ?
        """

    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute(sql, (flight_id_int,))
        rows = cur.fetchall()
    except Exception as e:
        return {"error": f"Error fetching flight info: {e}"}

    if not rows:
        return {"error": f"No flight found for flight_id={flight_id_int}"}

    # Find fares for this flight, and tickets left:
    sql_fares = """
        SELECT DISTINCT
            tf.fare_conditions,
            tf.amount AS price,
            COUNT(tf.ticket_no) AS tickets_left
        FROM ticket_flights AS tf
        WHERE tf.flight_id = ?
        GROUP BY tf.fare_conditions, tf.amount
        ORDER BY price ASC
        """
    try:
        cur.execute(sql_fares, (flight_id_int,))
        fares_rows = cur.fetchall()
    except Exception as e:
        return {"error": f"Error fetching fares: {e}"}
    finally:
        conn.close()

    # Build the flight object and attach fares
    # Rows may contain multiple fares; base flight metadata will be identical across rows
    (
        _fid,
        flight_no,
        scheduled_departure,
        scheduled_arrival,
        dep_airport,
        arr_airport,
        status,
        aircraft_code,
        actual_departure,
        actual_arrival,
    ) = rows[0]

    flight = {
        "flight_id": flight_id_int,
        "flight_no": flight_no,
        "scheduled_departure": str(scheduled_departure),
        "scheduled_arrival": str(scheduled_arrival),
        "departure_airport": dep_airport,
        "arrival_airport": arr_airport,
        "status": status,
        "aircraft_code": aircraft_code,
        "actual_departure": str(actual_departure),
        "actual_arrival": str(actual_arrival),
    }

    fares = []
    for (
        fare_conditions,
        price,
        tickets_left,
    ) in fares_rows:
        if tickets_left > 0:
            fares.append(
                {
                    "fare_conditions": str(fare_conditions),
                    "price": int(price),
                    "tickets_left": int(tickets_left),
                }
            )

    flight["fares"] = fares

    return {"flight_id": flight_id_int, "flight": flight}


@tool
def book_flight(
    flight_number: str,
    flight_datetime: list[datetime],
    fare_conditions: str,
    price: int,
    *,
    state: Annotated[Dict[str, Any], InjectedState],
    config: RunnableConfig,
):
    """
    Book a flight.
    Parameters:
    - flight_number: the flight number of the flight to book. e.g. "DY1126"
    - flight_datetime: the departure and arrival datetime of the flight to book. e.g. [2025-08-14 10:00:00, 2025-08-14 12:00:00]
    - fare_conditions: the fare conditions of the flight to book. e.g. "Economy", "Business" (case-sensitive)
    - price: the price of the flight to book. e.g. 100
    """
    # Basic validation & normalization
    if not isinstance(flight_number, str) or not flight_number.strip():
        return {"error": "flight_number must be a non-empty string like 'DY1126'."}
    if not isinstance(fare_conditions, str) or not fare_conditions.strip():
        return {
            "error": "fare_conditions must be a non-empty string (e.g., 'Economy')."
        }
    try:
        price_int = int(price)
    except Exception:
        return {"error": f"price must be an integer, got: {price!r}"}

    flight_number_norm = flight_number.strip().upper()
    fare_conditions_norm = fare_conditions.strip()

    # Coerce datetime input
    def _coerce_to_iso(dt_val):
        from datetime import datetime

        if isinstance(dt_val, datetime):
            return dt_val.isoformat(sep=" ")
        if isinstance(dt_val, str):
            s = dt_val.strip().replace("Z", "")
            try:
                return datetime.fromisoformat(s).isoformat(sep=" ")
            except Exception:
                # Accept as-is if already a reasonable string; DB can store TEXT
                return s
        return str(dt_val)

    # If flight_datetime is a list, coerce each item to ISO string and store as JSON string
    if isinstance(flight_datetime, (list, tuple)):
        flight_datetime_iso = json.dumps([_coerce_to_iso(dt) for dt in flight_datetime])
    else:
        flight_datetime_iso = _coerce_to_iso(flight_datetime)

    print(
        f"book_flight tool called with flight_number: {flight_number_norm}, flight_datetime: {flight_datetime_iso}, fare_conditions: {fare_conditions_norm}, price: {price_int}"
    )

    passenger_id = config.get("configurable", {}).get("passenger_id", None)
    flight_req_id = state.get("requirement_id")

    # Save booking to DB
    conn = None
    ticket_no = str(uuid.uuid4())
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ticket_flights (ticket_no, flight_id, fare_conditions, amount, flight_number, flight_datetime, flight_req_id, passenger_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticket_no,
                None,  # flight_id empty
                fare_conditions_norm,
                price_int,
                flight_number_norm,
                flight_datetime_iso,
                flight_req_id,
                passenger_id,
            ),
        )
        conn.commit()
    except Exception as e:
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return {"error": f"Booking failed: {e}"}
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

    # Prepare return data for LLM to read back
    return {
        "message": "Flight booked successfully.",
        "ticket_no": ticket_no,
        "flight_number": flight_number_norm,
        "flight_datetime": flight_datetime_iso,
        "fare_conditions": fare_conditions_norm,
        "price": price_int,
        "passenger_id": passenger_id,
        "flight_req_id": flight_req_id,
    }


@tool
def flight_search_handoff_tool(
    reason: Optional[str] = None,
    *,
    state: Annotated[Dict[str, Any], InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    A tool to mark the current task as completed and/or to handoff control of the dialog to the primary assistant,
    who can re-route the dialog based on the user's needs.
    Examples:
    - reason="Task completed: booked flight AC1234"
    - reason="User wants to book hotel instead"
    """

    return Command(
        update={
            "handoff": True,
            "handoff_reason": reason,
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Handoff: {reason}",
                )
            ],
        },
        goto=END,
    )
