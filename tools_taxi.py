from db_connection import db_file
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import sqlite3
import uuid
from datetime import datetime


@tool
def fetch_user_taxi_requests(config: RunnableConfig) -> list[dict]:
    """Fetch all taxi requests for the current logged in user.
    Returns:
        A list of dictionaries where each dictionary contains the taxi request details.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        # passenger_id = "3442 587242"
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM taxi_requests WHERE passenger_id = ?", (passenger_id,)
    )

    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return results


@tool
def create_taxi_request(
    pickup_location: str,
    dropoff_location: str,
    pickup_time: datetime,
    ride_status: str = "to_be_assigned",
    *,
    config: RunnableConfig,
) -> str:
    """Create a new taxi request for the current logged in user.
    You must know all the details (pickup_location, dropoff_location, pickup_time) of the taxi request before calling this tool.
    Returns:
        A string indicating the success of the request.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        # passenger_id = "3442 587242"
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO taxi_requests (request_id, passenger_id, pickup_location, dropoff_location, pickup_time, ride_status) VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            passenger_id,
            pickup_location,
            dropoff_location,
            pickup_time,
            ride_status,
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return "Taxi request created successfully."


@tool
def remove_taxi_request(request_id: str, *, config: RunnableConfig) -> str:
    """Remove a taxi request for the given request_id.
    Returns:
        A string indicating the success of the request.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        # passenger_id = "3442 587242"
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check if the request exists
    cursor.execute(
        "SELECT * FROM taxi_requests WHERE request_id = ? AND passenger_id = ?",
        (request_id, passenger_id),
    )
    row_found = cursor.fetchone()
    if not row_found:
        return "Taxi request not found."

    cursor.execute(
        "DELETE FROM taxi_requests WHERE request_id = ? AND passenger_id = ?",
        (request_id, passenger_id),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return "Taxi request removed successfully."


@tool
def update_taxi_request(
    request_id: str,
    pickup_location: str = None,
    dropoff_location: str = None,
    pickup_time: datetime = None,
    ride_status: str = None,
    *,
    config: RunnableConfig,
) -> str:
    """Update an existing taxi request for the given request_id.
    request_id must be a provided and be a valid request_id of the current logged in user.
    Any parameter can be provided and will be updated. At least one parameter must be provided.
    Returns:
        A string indicating the success of the request.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        # passenger_id = "3442 587242"
        raise ValueError("No passenger ID configured.")

    # Check if at least one parameter is provided for update
    if all(
        param is None
        for param in [pickup_location, dropoff_location, pickup_time, ride_status]
    ):
        return "No parameters provided for update. Please provide at least one parameter to update."

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check if the request exists
    cursor.execute(
        "SELECT * FROM taxi_requests WHERE request_id = ? AND passenger_id = ?",
        (request_id, passenger_id),
    )
    row_found = cursor.fetchone()
    if not row_found:
        cursor.close()
        conn.close()
        return f"Taxi request {request_id} not found."

    # Build the update query dynamically based on provided parameters
    param_map = [
        ("pickup_location", pickup_location),
        ("dropoff_location", dropoff_location),
        ("pickup_time", pickup_time),
        ("ride_status", ride_status),
    ]
    update_cols = []
    update_values = []
    for col, val in param_map:
        if val is not None:
            update_cols.append(f"{col} = ?")
            update_values.append(val)

    # Add request_id and passenger_id to the values for the WHERE clause
    update_values.extend([request_id, passenger_id])

    update_query = f"UPDATE taxi_requests SET {', '.join(update_cols)} WHERE request_id = ? AND passenger_id = ?"

    cursor.execute(update_query, update_values)
    conn.commit()
    cursor.close()
    conn.close()

    updated_fields = []
    if pickup_location is not None:
        updated_fields.append("pickup_location")
    if dropoff_location is not None:
        updated_fields.append("dropoff_location")
    if pickup_time is not None:
        updated_fields.append("pickup_time")
    if ride_status is not None:
        updated_fields.append("ride_status")

    return f"Taxi request {request_id} updated successfully. Updated fields: {', '.join(updated_fields)}."
