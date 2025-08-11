from datetime import datetime
from typing import Any, Dict, List, Optional
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
):
    """
    Search flights by exact departure/arrival airport match and a departure-time window.

    Parameters:
    departure_airport : str
        IATA/ICAO code, e.g. 'YUL', 'YHU'. Must match exactly.
    arrival_airport : str
        IATA/ICAO code, e.g. 'YYZ'. Must match exactly.
    departure_time : list[datetime]
        A two-item list [window_start, window_end]. Each item can be a `datetime` or an ISO 8601 string.
        Only flights with scheduled_departure within this window (inclusive) are returned.
    budget : Optional[int]
        Optional maximum ticket price (in the same currency/units as the DB's `amount`).
        If not provided (None), any price is accepted.

    Returns:
    dict
        A JSON-serializable dict with the applied filters and a list of matching flights. Each flight
        includes its fares (fare_conditions and amount). Intended for direct LLM tool consumption.
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

    # --- build SQL ---
    sql = """
        SELECT
            f.flight_id,
            f.flight_no,
            f.scheduled_departure,
            f.scheduled_arrival,
            f.departure_airport,
            f.arrival_airport,
            tf.fare_conditions,
            tf.amount
        FROM flights AS f
        JOIN (
            SELECT DISTINCT flight_id, fare_conditions, amount
            FROM ticket_flights
        ) AS tf ON tf.flight_id = f.flight_id
        WHERE
            (f.status NOT IN ('Arrived','Cancelled','Departed')) AND
            f.departure_airport = ? AND
            f.arrival_airport = ? AND
            f.scheduled_departure >= ? AND
            f.scheduled_departure <= ?
        """

    params: list[Any] = [
        departure_airport,
        arrival_airport,
        window_start.strftime("%Y-%m-%d %H:%M:%S"),
        window_end.strftime("%Y-%m-%d %H:%M:%S"),
    ]

    if budget is not None:
        sql += " AND tf.amount <= ?"
        params.append(int(budget))

    sql += " ORDER BY f.scheduled_departure ASC, tf.amount ASC"

    # --- execute query ---
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        conn.close()

    # --- group rows by flight ---
    flights_map: Dict[int, Dict[str, Any]] = {}
    for (
        flight_id,
        flight_no,
        scheduled_departure,
        scheduled_arrival,
        dep_airport,
        arr_airport,
        fare_conditions,
        amount,
    ) in rows:
        if flight_id not in flights_map:
            flights_map[flight_id] = {
                "flight_id": int(flight_id),
                "flight_no": flight_no,
                # SQLite returns strings for TIMESTAMP; cast to str to be safe
                "scheduled_departure": str(scheduled_departure),
                "scheduled_arrival": str(scheduled_arrival),
                "departure_airport": dep_airport,
                "arrival_airport": arr_airport,
                "fares": [],
            }
        flights_map[flight_id]["fares"].append(
            {"fare_conditions": str(fare_conditions), "amount": int(amount)}
        )

    results: List[Dict[str, Any]] = list(flights_map.values())

    # convenience: add minAmount per flight for quick sorting/selection by the agent
    for item in results:
        fares = item.get("fares", [])
        item["min_amount"] = min((f["amount"] for f in fares), default=None)

    return {
        "filters_applied": {
            "departure_airport": departure_airport,
            "arrival_airport": arrival_airport,
            "departure_time_window": [
                window_start.isoformat(sep=" "),
                window_end.isoformat(sep=" "),
            ],
            "budget": budget,
        },
        "results": results,
    }


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
    flight_id: int,
    fare_conditions: str,
    *,
    state: Annotated[Dict[str, Any], InjectedState],
    config: RunnableConfig,
):
    """
    Book a flight for the current user.
    Parameters:
    flight_id : int
        The flight_id of the flight to book.
    fare_conditions : str
        The fare_conditions of the flight to book(e.g. 'Economy', 'Business'), case insensitive.

    Returns:
    dict
        A JSON-serializable dict with the booking information.
    """
    # --- validate inputs ---
    try:
        flight_id = int(flight_id)
    except Exception:
        return {"error": f"flight_id must be an integer, got: {flight_id!r}"}

    if not isinstance(fare_conditions, str) or not fare_conditions.strip():
        return {"error": "fare_conditions must be a non-empty string."}

    passenger_id = config.get("configurable", {}).get("passenger_id", None)
    if not passenger_id:
        return {"error": "Missing `passenger_id` in state; cannot complete booking."}
    fare_conditions_norm = fare_conditions.strip()

    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()

        # Step 2: select a random available ticket for this flight+fare
        cur.execute(
            """
            SELECT
                tf.ticket_no,
                tf.fare_conditions,
                tf.amount,
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
            FROM ticket_flights AS tf
            JOIN flights AS f ON f.flight_id = tf.flight_id
            WHERE tf.flight_id = ?
              AND tf.fare_conditions = ? COLLATE NOCASE
            ORDER BY RANDOM()
            LIMIT 1
            """,
            (flight_id, fare_conditions_norm),
        )
        row = cur.fetchone()

        # If no available ticket found for the given flight_id and fare_conditions
        if not row:
            # Clarify whether the flight exists for better error messaging
            cur.execute(
                "SELECT 1 FROM flights WHERE flight_id = ? LIMIT 1", (flight_id,)
            )
            exists = cur.fetchone() is not None
            if not exists:
                conn.rollback()
                return {"error": f"Flight not found for flight_id={flight_id}."}
            conn.rollback()
            return {
                "error": f"No available tickets for flight_id={flight_id} in fare_conditions={fare_conditions_norm!r}."
            }

        (
            ticket_no,
            fare_cond_db,
            amount,
            f_fid,
            flight_no,
            scheduled_departure,
            scheduled_arrival,
            dep_airport,
            arr_airport,
            status,
            aircraft_code,
            actual_departure,
            actual_arrival,
        ) = row

        # make the booking: sync the passenger_id to the 'tickets' table
        cur.execute(
            "SELECT ticket_no, book_ref, passenger_id FROM tickets WHERE ticket_no = ?",
            (ticket_no,),
        )
        ticket_row = cur.fetchone()

        created = False
        previous_passenger_id = None
        if not ticket_row:
            book_ref = f"BR-{uuid.uuid4().hex[:8].upper()}"
            cur.execute(
                "INSERT INTO tickets (ticket_no, book_ref, passenger_id) VALUES (?, ?, ?)",
                (ticket_no, book_ref, passenger_id),
            )
            created = True
        else:
            _tno, book_ref, prev_pid = ticket_row
            previous_passenger_id = prev_pid
            cur.execute(
                "UPDATE tickets SET passenger_id = ? WHERE ticket_no = ?",
                (passenger_id, ticket_no),
            )

        conn.commit()

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"error": f"Booking failed: {e}"}
    finally:
        conn.close()

    flight_obj = {
        "flight_id": int(f_fid),
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

    return {
        "success": True,
        "created_ticket_record": created,
        # "previous_passenger_id": previous_passenger_id,
        "ticket": {
            "ticket_no": ticket_no,
            "book_ref": book_ref,
            "passenger_id": passenger_id,
        },
        "fare": {
            "fare_conditions": str(fare_cond_db),
            "amount": int(amount),
        },
        "flight": flight_obj,
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
    Workflow:
    - Worker calls this tool when the flight task is done or needs escalation.
    - This tool returns state keys `handoff=True` and `handoff_reason`.
    - The supervisor wrapper inspects the worker output and pops out of worker mode.
    Examples:
    - reason="Task completed: booked flight 12345"
    - reason="Need supervisor to confirm next steps"
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
