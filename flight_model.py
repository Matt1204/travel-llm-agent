from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, field_validator, ConfigDict, Field


def _to_datetime(value: Union[str, datetime]) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        v = value.replace("Z", "")
        return datetime.fromisoformat(v)
    raise TypeError(f"Unsupported datetime value: {value!r}")

# ----------------------
# Builders & Normalizers to ease transition from DB tools
# ----------------------
class SegmentModel(BaseModel):
    flight_number: Optional[str] = None
    carrier_code: Optional[str] = None
    number: Optional[str] = None
    departure_airport: Optional[str] = None
    departure_at: datetime
    arrival_airport: Optional[str] = None
    arrival_at: datetime

    model_config = ConfigDict(extra="ignore")

    @field_validator("departure_at", "arrival_at", mode="before")
    @classmethod
    def _parse_dt(cls, v):
        try:
            return _to_datetime(v)
        except Exception:
            return v

class FareModel(BaseModel):
    fare_conditions: str
    amount: int

    model_config = ConfigDict(extra="ignore")

class FlightOfferModel(BaseModel):
    flight_no: Optional[str] = None
    flight_numbers: List[str] = []
    scheduled_departure: datetime
    scheduled_arrival: datetime
    departure_airport: Optional[str] = None
    arrival_airport: Optional[str] = None
    segments: List[SegmentModel]
    fares: List[FareModel]
    is_direct: bool
    min_amount: Optional[int] = None
    # Keep space for raw but exclude it from dumps to keep tool outputs tidy
    # raw: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="ignore")

    @field_validator("scheduled_departure", "scheduled_arrival", mode="before")
    @classmethod
    def _parse_sched_dt(cls, v):
        try:
            return _to_datetime(v)
        except Exception:
            return v

    @property
    def connections(self) -> int:
        return max(0, len(self.segments) - 1)