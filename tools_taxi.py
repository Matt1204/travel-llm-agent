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


# create_taxi_request.invoke(
#     {
#         "pickup_location": "1445 Guy St, Montreal, Quebec H3H 2L5",
#         "dropoff_location": "110 Notre-Dame St W, Montreal, Quebec H2Y 1T1",
#         "pickup_time": datetime(2025, 7, 28, 13, 30, 0),
#     }
# )
# fetch_user_taxi_requests.invoke({})

# remove_taxi_request.invoke(
#     {
#         "request_id": "0d73c806-3ec1-4b1f-a1bc-0e7b6bfa4960",
#     }
# )
