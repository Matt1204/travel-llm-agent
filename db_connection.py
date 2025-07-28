import sqlite3
from datetime import datetime
import uuid
import pandas as pd

# Connect to local SQLite database file (same directory)
# db_file = "llm-agent-db.db"
# Convert the flights to present time for our tutorial
def update_dates(file):
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    return file


# db_file = update_dates("llm-agent-db.db")
db_file = "llm-agent-db.db"
# db_file = "llm-agent-db.db"
# conn = sqlite3.connect(db)
# cursor = conn.cursor()


# -------------------------------
# INSERT operations
# -------------------------------
# Insert single row
# cursor.execute(
#     "INSERT INTO taxi_requests (request_id, passenger_id, pickup_location, dropoff_location, pickup_time, ride_status) VALUES (?, ?, ?, ?, ?, ?)",
#     (
#         str(uuid.uuid4()),
#         "3442 587242",
#         "1445 Guy St, Montreal, Quebec H3H 2L5",
#         "110 Notre-Dame St W, Montreal, Quebec H2Y 1T1",
#         datetime(2025, 7, 28, 18, 7, 0),
#         "to_be_assigned",
#     ),
# )
# conn.commit()
# conn.close()
