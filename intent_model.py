from typing import Dict, Any, List, Union, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import json
from db_connection import db_file
import sqlite3
import uuid

class PriorityRequirement(BaseModel):
    """Base class for priority-based requirements with dynamic priority levels."""

    def __init__(self, **data):
        super().__init__(**data)
        # Store all priority data in a private dict for easy manipulation
        self._priorities = {k: v for k, v in data.items() if k.startswith("priority_")}

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name.startswith("priority_"):
            if not hasattr(self, "_priorities"):
                self._priorities = {}
            self._priorities[name] = value

    def add_priority(self, level: int, value: Any) -> None:
        """Add a new priority level."""
        priority_key = f"priority_{level}"
        setattr(self, priority_key, value)
        self._priorities[priority_key] = value

    def remove_priority(self, level: int) -> None:
        """Remove a priority level."""
        priority_key = f"priority_{level}"
        if hasattr(self, priority_key):
            delattr(self, priority_key)
            self._priorities.pop(priority_key, None)

    def update_priority(self, level: int, value: Any) -> None:
        """Update an existing priority level."""
        priority_key = f"priority_{level}"
        if hasattr(self, priority_key):
            setattr(self, priority_key, value)
            self._priorities[priority_key] = value
        else:
            raise ValueError(f"Priority level {level} does not exist")

    def get_priority(self, level: int) -> Any:
        """Get value for a specific priority level."""
        priority_key = f"priority_{level}"
        return getattr(self, priority_key, None)

    def get_all_priorities(self) -> Dict[str, Any]:
        """Get all priority levels as a dictionary."""
        return dict(self._priorities)

    def get_sorted_priorities(self) -> List[tuple]:
        """Get priorities sorted by level (priority_1, priority_2, etc.)."""
        sorted_items = sorted(
            self._priorities.items(), key=lambda x: int(x[0].split("_")[1])
        )
        return [(int(k.split("_")[1]), v) for k, v in sorted_items]

    def get_highest_priority(self) -> Any:
        """Get the highest priority value (priority_1)."""
        return self.get_priority(1)

    def has_priority(self, level: int) -> bool:
        """Check if a priority level exists."""
        return hasattr(self, f"priority_{level}")

    def get_priority_levels(self) -> List[int]:
        """Get all available priority levels as sorted integers."""
        return sorted([int(k.split("_")[1]) for k in self._priorities.keys()])


PriorityRequirement.model_config = {"extra": "allow"}


class AirportRequirements(PriorityRequirement):
    """Airport requirement with priority levels."""

    @field_validator("*", mode="before")
    def ensure_string(cls, v):
        """Ensure all priority values are strings."""
        if isinstance(v, str):
            return v
        else:
            raise ValueError("Airport requirement must be a string")


class TimeRequirement(PriorityRequirement):
    """Time requirement with priority levels represented as [start, end] datetimes."""

    @field_validator("*", mode="before")
    def ensure_list_of_datetimes(cls, v):
        if isinstance(v, list):
            return [datetime.fromisoformat(item) for item in v]
        return [datetime.fromisoformat(v)]

    @field_validator("*", mode="before")
    def validate_time_windows(cls, v):
        if isinstance(v, list) and len(v) == 2:
            start, end = v
            if end <= start:
                raise ValueError("End time must be after start time")
            return [start, end]
        raise ValueError("Time requirement must be a list: [start, end]")


class BudgetRequirement(PriorityRequirement):
    """Budget requirement with priority levels."""

    @field_validator("*", mode="before")
    def ensure_list_of_ints(cls, v):
        if v is None:
            return []

        if isinstance(v, list):
            return [int(item) for item in v]

    @field_validator("*", mode="before")
    def validate_positive_budget(cls, v):
        if isinstance(v, (int, float)) and v <= 0:
            raise ValueError("Budget must be positive")
        return v

    @field_validator("*", mode="before")
    def validate_budget_boundaries(cls, v):
        if isinstance(v, list) and len(v) == 2:
            min_budget, max_budget = v
            if min_budget >= max_budget:
                raise ValueError("Min budget must be less than max budget")
            return [min_budget, max_budget]
        raise ValueError("Budget requirement must be a list: [min_budget, max_budget]")


class FlightRequirements(BaseModel):
    """Main flight requirements class with dynamic priority-based attributes."""

    departure_airport: Optional[AirportRequirements] = None
    arrival_airport: Optional[AirportRequirements] = None
    departure_time: Optional[TimeRequirement] = None
    # return_time: Optional[TimeRequirement] = None
    budget: Optional[BudgetRequirement] = None

    def __init__(self, **data):
        # If no data provided, create empty objects with default priority_1 levels
        if not data:
            data = {
                "departure_airport": {"priority_1": None},
                "arrival_airport": {"priority_1": None},
                "departure_time": {"priority_1": None},
                # "return_time": {"priority_1": None},
                "budget": {"priority_1": None},
            }

        # Convert nested dicts to appropriate requirement objects
        processed_data = {}

        # wrapping init values to requirement objects
        for key, value in data.items():
            if key in ["departure_airport", "arrival_airport"] and isinstance(
                value, dict
            ):
                processed_data[key] = AirportRequirements(**value)
            elif key in ["departure_time"] and isinstance(value, dict):
                processed_data[key] = TimeRequirement(**value)
            elif key == "budget" and isinstance(value, dict):
                processed_data[key] = BudgetRequirement(**value)
            else:
                processed_data[key] = value

        super().__init__(**processed_data)

    def add_requirement_priority(
        self, requirement_type: str, level: int, value: Any
    ) -> None:
        """Add a priority level to a specific requirement type."""
        if not hasattr(self, requirement_type):
            raise ValueError(f"Unknown requirement type: {requirement_type}")

        requirement = getattr(self, requirement_type)
        if requirement is None:
            # Create new requirement object
            if requirement_type in ["departure_airport", "arrival_airport"]:
                requirement = AirportRequirements()
            elif requirement_type in ["departure_time"]:
                requirement = TimeRequirement()
            elif requirement_type == "budget":
                requirement = BudgetRequirement()
            else:
                requirement = PriorityRequirement()

            setattr(self, requirement_type, requirement)

        requirement.add_priority(level, value)

    def get_requirement_obj(self, requirement_type: str) -> Any:
        """Get a specific requirement object."""
        return getattr(self, requirement_type, None)

    def remove_requirement_priority(self, requirement_type: str, level: int) -> None:
        """Remove a priority level from a specific requirement type."""
        requirement = getattr(self, requirement_type, None)
        if requirement:
            requirement.remove_priority(level)

    def update_requirement_priority(
        self, requirement_type: str, level: int, value: Any
    ) -> None:
        """Update a priority level for a specific requirement type."""
        requirement = getattr(self, requirement_type, None)
        if requirement:
            requirement.update_priority(level, value)
        else:
            raise ValueError(f"Requirement type {requirement_type} does not exist")

    def get_requirement_priority(self, requirement_type: str, level: int) -> Any:
        """Get a specific priority level from a requirement type."""
        requirement = getattr(self, requirement_type, None)
        return requirement.get_priority(level) if requirement else None

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary matching the original structure."""
        result = {}

        for field_name in self.__fields__:
            requirement = getattr(self, field_name)
            if requirement is not None:
                result[field_name] = requirement.get_all_priorities()

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_json_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_data: Union[str, Dict[str, Any]]) -> "FlightRequirements":
        """Create FlightRequirements from JSON data."""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        return cls(**data)


# Usage examples and helper functions
def create_sample_requirements() -> FlightRequirements:
    """Create sample flight requirements for testing."""
    sample_data = {
        "departure_airport": {
            "priority_1": "JFK",
            "priority_2": "LGA",
            "priority_3": "EWR",
            "priority_4": "HPN",
        },
        "arrival_airport": {
            "priority_1": "LAX",
            "priority_2": "BUR",
            "priority_3": "SNA",
        },
        "departure_time": {
            "priority_1": [
                datetime(2025, 8, 1, 0, 0),
                datetime(2025, 8, 2, 23, 59, 59),
            ],
            "priority_2": [
                datetime(2025, 8, 1, 0, 0),
                datetime(2025, 8, 7, 23, 59, 59),
            ],
        },
        "budget": {
            "priority_1": [1000, 1200],
            "priority_2": [1200, 1500],
            "priority_3": [1500, 2000],
            "priority_4": [2000, 2500],
        },
    }

    return FlightRequirements(**sample_data)



def save_flight_requirements_to_db(
    requirements: FlightRequirements, passenger_id: str
) -> None:
    """Save flight requirements to the database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    id = str(uuid.uuid4())

    arrival_airport_json = json.dumps(requirements.get_requirement_obj("arrival_airport").get_all_priorities())

    departure_airport_json = json.dumps(requirements.get_requirement_obj("departure_airport").get_all_priorities())

    budget_json = json.dumps(requirements.get_requirement_obj("budget").get_all_priorities())

    departure_time = requirements.get_requirement_obj("departure_time").get_all_priorities()
    departure_time_json = json.dumps([[dt.isoformat() for dt in value] for value in departure_time.values()])

    cursor.execute(
        "INSERT INTO flight_requirements (requirement_id, passenger_id, departure_airport, arrival_airport, departure_time, budget) VALUES (?, ?, ?, ?, ?, ?)",
        (
            id,
            passenger_id,
            departure_airport_json,
            arrival_airport_json,
            departure_time_json,
            budget_json,
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()

def fetch_requirements_from_db(requirement_id: str) -> None:
    """Fetch the flight requirements from the database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flight_requirements WHERE requirement_id = ?", (requirement_id,))
    result = cursor.fetchone()
    passenger_id = result[1]
    departure_airport = json.loads(result[2])
    arrival_airport = json.loads(result[3])
    departure_time = json.loads(result[4])
    budget = json.loads(result[5])
    print(passenger_id)
    print(departure_airport)
    print(arrival_airport)

# Example usage
if __name__ == "__main__":
    # Create empty requirements (will have default priority_1 levels)
    reqs = create_sample_requirements()
    # save_flight_requirements_to_db(reqs, "1234567890")
    fetch_requirements_from_db("186e7310-08d8-4d3b-984a-2e4d13910121")

    print("=== Empty FlightRequirements ===")
    print(reqs.to_json())

    print("\n=== Adding new priority level ===")
    reqs.add_requirement_priority("departure_airport", 1, ["LAX"])
    print(reqs.to_json())

    print("\n=== Getting priority level ===")
    print(reqs.get_requirement_priority("departure_airport", 1))

    print("\n=== Updating existing priority ===")
    reqs.update_requirement_priority("departure_airport", 1, ["LAX", "JFK"])
    print(reqs.to_json())

    print("\n=== Getting sorted priorities for budget ===")
    budget_priorities = reqs.budget.get_sorted_priorities()
    for level, value in budget_priorities:
        print(f"Priority {level}: ${value}")

    print("\n=== Removing a priority level ===")
    reqs.remove_requirement_priority("budget", 4)
    print(f"Budget priority levels after removal: {reqs.budget.get_priority_levels()}")

    print("\n=== Final JSON output ===")
    print(reqs.to_json())
