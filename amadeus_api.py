from __future__ import annotations

import os
import time
from dataclasses import dataclass
import json
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

try:
    # Optional: import the intent model if available, for type hints and helpers
    from intent_model import FlightRequirements
except Exception:  # pragma: no cover - keep import optional
    FlightRequirements = Any  # type: ignore


AMADEUS_OAUTH_URL = {
    "test": "https://test.api.amadeus.com/v1/security/oauth2/token",
    "production": "https://api.amadeus.com/v1/security/oauth2/token",
}

AMADEUS_BASE_URL = {
    "test": "https://test.api.amadeus.com",
    "production": "https://api.amadeus.com",
}


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None else default


def _to_datetime(value: Union[str, datetime]) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        v = value.replace("Z", "")
        return datetime.fromisoformat(v)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _daterange(start_date: date, end_date: date) -> List[date]:
    days = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(days + 1)]


@dataclass
class OAuthToken:
    access_token: str
    expires_at_epoch: float

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "OAuthToken":
        # Amadeus returns expires_in in seconds
        expires_in = int(data.get("expires_in", 0))
        # Renew slightly earlier to avoid edge expiry
        expires_at = time.time() + max(0, expires_in - 30)
        return cls(access_token=str(data["access_token"]), expires_at_epoch=expires_at)

    def is_valid(self) -> bool:
        return time.time() < self.expires_at_epoch


class AmadeusClient:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        environment: str = "test",
        base_url: Optional[str] = None,
        timeout_seconds: int = 20,
    ) -> None:
        self.client_id = client_id or _env("AMADEUS_CLIENT_ID")
        self.client_secret = client_secret or _env("AMADEUS_CLIENT_SECRET")
        self.environment = environment or _env("AMADEUS_ENV", "test") or "test"
        self.base_url = (
            base_url
            or _env("AMADEUS_BASE_URL")
            or AMADEUS_BASE_URL.get(self.environment, AMADEUS_BASE_URL["test"])
        )

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Missing Amadeus credentials. Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET."
            )

        self._oauth_url = AMADEUS_OAUTH_URL.get(
            self.environment, AMADEUS_OAUTH_URL["test"]
        )
        self._token: Optional[OAuthToken] = None
        self._timeout = timeout_seconds

        self._session = requests.Session()

    # ----------------------
    # OAuth
    # ----------------------
    def _ensure_token(self) -> str:
        if self._token and self._token.is_valid():
            return self._token.access_token

        resp = self._session.post(
            self._oauth_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Amadeus OAuth failed: {resp.status_code} {resp.text[:500]}"
            )
        data = resp.json()
        self._token = OAuthToken.from_response(data)
        return self._token.access_token

    def _auth_headers(self) -> Dict[str, str]:
        token = self._ensure_token()
        return {"Authorization": f"Bearer {token}"}

    # ----------------------
    # Flight Offers Search (GET, per-day query)
    # ----------------------
    def search_flight_offers_by_day(
        self,
        origin: str,
        destination: str,
        departure_date: Union[str, date],
        *,
        adults: int = 1,
        currency: Optional[str] = None,
        non_stop: Optional[bool] = None,
        max_results: Optional[int] = None,
        included_airline_code: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query Flight Offers Search for a single departure day using the public GET interface.

        Docs: GET /v2/shopping/flight-offers
        Reference fields used: originLocationCode, destinationLocationCode, departureDate, adults, currencyCode, max, nonStop
        """
        if isinstance(departure_date, date):
            dep_str = departure_date.isoformat()
        else:
            dep_str = departure_date

        params: Dict[str, Any] = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": dep_str,
            "adults": int(adults),
        }
        if currency:
            params["currencyCode"] = currency
        if non_stop is not None:
            params["nonStop"] = str(bool(non_stop)).lower()
        if max_results is not None:
            params["max"] = int(max_results)
        if included_airline_code:
            params["includedAirlineCodes"] = included_airline_code
        if additional_params:
            params.update(additional_params)

        url = f"{self.base_url}/v2/shopping/flight-offers"
        resp = self._session.get(
            url, headers=self._auth_headers(), params=params, timeout=self._timeout
        )
        if resp.status_code >= 400:
            return {
                "error": True,
                "status": resp.status_code,
                "message": resp.text,
            }
        return resp.json()

    # ----------------------
    # Flight Offers Search across a time window (aggregates multiple days)
    # ----------------------
    def amadeus_search_flights(
        self,
        origin: str,
        destination: str,
        departure_time_window: List[Union[str, datetime]],
        *,
        adults: int = 1,
        currency: Optional[str] = None,
        non_stop: Optional[bool] = None,
        max_results_per_day: int = 50,
        included_airline_code: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience helper that accepts a [start, end] datetime window and aggregates results
        by performing one GET search call per day. This mirrors the DB-based time-window search.
        """
        if (
            not isinstance(departure_time_window, (list, tuple))
            or len(departure_time_window) != 2
        ):
            return {
                "error": True,
                "message": "departure_time_window must be [start, end]",
            }

        start_dt = _to_datetime(departure_time_window[0])
        end_dt = _to_datetime(departure_time_window[1])
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        days = _daterange(start_dt.date(), end_dt.date())
        aggregated: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        last_meta: Dict[str, Any] = {}
        last_warnings: List[Any] = []

        for d in days:
            res = self.search_flight_offers_by_day(
                origin=origin,
                destination=destination,
                departure_date=d,
                adults=adults,
                currency=currency,
                non_stop=non_stop,
                max_results=max_results_per_day,
                included_airline_code=included_airline_code,
                additional_params=additional_params,
            )
            if isinstance(res, dict) and res.get("error"):
                # Return the first error encountered
                return res

            data = (res or {}).get("data") or []
            meta = (res or {}).get("meta") or {}
            warnings = (res or {}).get("warnings") or []
            last_meta = meta or last_meta
            last_warnings = warnings or last_warnings

            for offer in data:
                # oid = str(offer.get("id", ""))
                # if oid and oid in seen_ids:
                #     continue
                # if oid:
                #     seen_ids.add(oid)
                # Exclude the "id" entry in every offer dict
                offer_no_id = {k: v for k, v in offer.items() if k != "id"}
                aggregated.append(offer_no_id)

        # return {"data": aggregated, "meta": last_meta, "warnings": last_warnings}
        return aggregated

    # ----------------------
    # Flight Offers Price (POST)
    # ----------------------
    def price_flight_offers(
        self,
        flight_offers: List[Dict[str, Any]],
        *,
        include: Optional[List[str]] = None,
        force_class: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Confirm pricing for given flight offers.

        Docs: POST /v1/shopping/flight-offers/pricing
        """
        body: Dict[str, Any] = {
            "data": {
                "type": "flight-offers-pricing",
                "flightOffers": flight_offers,
            }
        }
        params: Dict[str, Any] = {}
        if include:
            # CSV list per OpenAPI docs (credit-card-fees,bags,other-services,detailed-fare-rules)
            params["include"] = ",".join(include)
        if force_class is not None:
            params["forceClass"] = str(bool(force_class)).lower()

        url = f"{self.base_url}/v1/shopping/flight-offers/pricing"
        headers = {**self._auth_headers(), "Content-Type": "application/json"}
        resp = self._session.post(
            url, headers=headers, params=params, json=body, timeout=self._timeout
        )
        if resp.status_code >= 400:
            return {
                "error": True,
                "status": resp.status_code,
                "message": resp.text,
            }
        return resp.json()


# ----------------------
# Builders & Normalizers to ease transition from DB tools
# ----------------------
def select_highest_priority(requirement_obj: Any) -> Optional[Any]:
    if requirement_obj is None:
        return None
    try:
        # The custom PriorityRequirement API
        for level, value in requirement_obj.get_sorted_priorities():
            if value is not None:
                return value
    except Exception:
        # Fallback for plain dicts
        if isinstance(requirement_obj, dict):
            # pick the lowest priority index (priority_1 preferred)
            keys = sorted(
                [k for k in requirement_obj.keys() if str(k).startswith("priority_")],
                key=lambda k: int(str(k).split("_")[1]),
            )
            for k in keys:
                v = requirement_obj.get(k)
                if v is not None:
                    return v
    return None


def build_search_from_requirements(
    requirements: FlightRequirements,
) -> Tuple[Optional[str], Optional[str], Optional[List[datetime]], Optional[int]]:
    """
    Extract a single best set of inputs from `FlightRequirements`:
    - departure_airport (IATA)
    - arrival_airport (IATA)
    - departure_time_window [start, end]
    - budget_max (highest-allowed price if a range exists)
    """
    dep = select_highest_priority(getattr(requirements, "departure_airport", None))
    arr = select_highest_priority(getattr(requirements, "arrival_airport", None))
    time_window = select_highest_priority(getattr(requirements, "departure_time", None))
    budget_window = select_highest_priority(getattr(requirements, "budget", None))

    dep_code = None
    arr_code = None
    if isinstance(dep, list):
        dep_code = dep[0]
    elif isinstance(dep, str):
        dep_code = dep

    if isinstance(arr, list):
        arr_code = arr[0]
    elif isinstance(arr, str):
        arr_code = arr

    dt_window: Optional[List[datetime]] = None
    if isinstance(time_window, list) and len(time_window) == 2:
        start = _to_datetime(time_window[0])
        end = _to_datetime(time_window[1])
        dt_window = [start, end]

    budget_max: Optional[int] = None
    if isinstance(budget_window, list) and len(budget_window) == 2:
        # [min, max]
        try:
            budget_max = int(budget_window[1])
        except Exception:
            budget_max = None

    return dep_code, arr_code, dt_window, budget_max


def normalize_offers_for_tool(offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Amadeus flight-offer objects to a simplified structure reminiscent of the
    DB-backed `search_flights` tool output (not 1:1, but close enough to adapt quickly).

    Improvements:
    - Correctly handle N segments across 1..N itineraries by scanning all segments.
    - Compute earliest departure and latest arrival across all segments.
    - Return a `segments` array with per-segment details, and a `flight_numbers` list.
    - Preserve `flight_no` (first segment's flight number) for backward compatibility.
    - Attach the raw offer under `raw` for downstream ranking and filtering.

    Review note (potentially error-prone):
    - Pricing uses `grandTotal` (fallback to `total`) and rounds to int. If exact cents or
      fees matter, use the pricing API or keep the original decimal string.
    - Cabin inference: read from the first traveler's first segment; mixed-cabin offers
      may be misrepresented.
    """
    normalized: List[Dict[str, Any]] = []
    for offer in offers:
        itineraries = offer.get("itineraries") or []
        if not itineraries:
            continue

        # Collect all segments across all itineraries in order
        all_segments: List[Dict[str, Any]] = []
        for itin in itineraries:
            segs = itin.get("segments") or []
            for seg in segs:
                if isinstance(seg, dict):
                    all_segments.append(seg)

        if not all_segments:
            continue

        # Earliest departure and latest arrival across ALL segments
        def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
            cur: Any = d
            for k in keys:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(k)
            return cur

        def _to_dt_str(val: Any) -> Optional[str]:
            if not val:
                return None
            s = str(val).replace("Z", "")
            try:
                # ensure it's a valid isoformat; return standardized string
                from datetime import datetime as _dt

                return _dt.fromisoformat(s).isoformat(sep=" ")
            except Exception:
                # keep original if not strictly isoformat-parsable
                return str(val)

        # Determine first (earliest) departure and last (latest) arrival
        dep_candidates: List[Tuple[str, str]] = []  # (iata, at)
        arr_candidates: List[Tuple[str, str]] = []

        for seg in all_segments:
            dep_iata = _safe_get(seg, "departure", "iataCode")
            dep_at = _to_dt_str(_safe_get(seg, "departure", "at"))
            arr_iata = _safe_get(seg, "arrival", "iataCode")
            arr_at = _to_dt_str(_safe_get(seg, "arrival", "at"))
            if dep_iata and dep_at:
                dep_candidates.append((dep_iata, dep_at))
            if arr_iata and arr_at:
                arr_candidates.append((arr_iata, arr_at))

        # Fallbacks if parsing failed
        if dep_candidates:
            # earliest by timestamp-like string; attempt datetime ordering
            try:
                from datetime import datetime as _dt

                dep_candidates.sort(
                    key=lambda x: _dt.fromisoformat(x[1].replace("Z", ""))
                )
            except Exception:
                dep_candidates.sort(key=lambda x: x[1])
            dep_airport, dep_time = dep_candidates[0]
        else:
            # fallback to first segment raw values
            first_seg = all_segments[0]
            dep_airport = _safe_get(first_seg, "departure", "iataCode")
            dep_time = _safe_get(first_seg, "departure", "at")

        if arr_candidates:
            try:
                from datetime import datetime as _dt

                arr_candidates.sort(
                    key=lambda x: _dt.fromisoformat(x[1].replace("Z", ""))
                )
            except Exception:
                arr_candidates.sort(key=lambda x: x[1])
            arr_airport, arr_time = arr_candidates[-1]
        else:
            last_seg = all_segments[-1]
            arr_airport = _safe_get(last_seg, "arrival", "iataCode")
            arr_time = _safe_get(last_seg, "arrival", "at")

        # Build per-segment details + flight numbers
        segment_details: List[Dict[str, Any]] = []
        flight_numbers: List[str] = []
        for seg in all_segments:
            carrier = (seg.get("carrierCode") or "").strip()
            number = str(seg.get("number") or "").strip()
            fno = f"{carrier}{number}" if (carrier or number) else None
            if fno:
                flight_numbers.append(fno)

            segment_details.append(
                {
                    "flight_number": fno,
                    "carrier_code": carrier or None,
                    "number": number or None,
                    "departure_airport": _safe_get(seg, "departure", "iataCode"),
                    "departure_at": _to_dt_str(_safe_get(seg, "departure", "at")),
                    "arrival_airport": _safe_get(seg, "arrival", "iataCode"),
                    "arrival_at": _to_dt_str(_safe_get(seg, "arrival", "at")),
                }
            )

        # Back-compat: primary flight_no is the first segment's number if present
        primary_flight_no = flight_numbers[0] if flight_numbers else None

        # Price
        price = (offer.get("price") or {}).get("grandTotal") or (
            offer.get("price") or {}
        ).get("total")
        try:
            amount_int = int(round(float(price))) if price is not None else None
        except Exception:
            amount_int = None

        # best-effort cabin from fareDetails
        traveler_pricings = offer.get("travelerPricings") or []
        cabin = None
        if traveler_pricings:
            fd = traveler_pricings[0].get("fareDetailsBySegment") or []
            if fd:
                cabin = fd[0].get("cabin")

        normalized.append(
            {
                "flight_no": primary_flight_no,
                "flight_numbers": flight_numbers,
                "scheduled_departure": dep_time,
                "scheduled_arrival": arr_time,
                "departure_airport": dep_airport,
                "arrival_airport": arr_airport,
                "segments": segment_details,
                "fares": [
                    {
                        "fare_conditions": str(cabin or "UNKNOWN"),
                        "amount": amount_int if amount_int is not None else 0,
                    }
                ],
                "is_direct": len(all_segments) == 1,
                "min_amount": amount_int if amount_int is not None else None,
            }
        )

    return normalized


# ----------------------
# Ranking utilities (price, duration, stops)
# ----------------------
def _parse_iso8601_duration_to_minutes(value: Optional[str]) -> Optional[int]:
    """
    Convert ISO8601 duration like "PT9H10M", "PT3H", "PT45M", or "P1DT2H" to minutes.
    Returns None if parsing fails.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        # Very small parser for common patterns: PnDTnHnM
        s = value.upper().strip()
        if not s.startswith("P"):
            return None
        days = hours = minutes = 0
        # Split date and time parts
        if "T" in s:
            date_part, time_part = s[1:].split("T", 1)
        else:
            date_part, time_part = s[1:], ""

        # Days
        if "D" in date_part:
            d_str, remainder = date_part.split("D", 1)
            days = int(d_str) if d_str else 0
        # Hours and minutes
        if time_part:
            # Hours
            if "H" in time_part:
                h_str, time_part = time_part.split("H", 1)
                hours = int(h_str) if h_str else 0
            # Minutes
            if "M" in time_part:
                m_str, _ = time_part.split("M", 1)
                minutes = int(m_str) if m_str else 0

        total_minutes = days * 24 * 60 + hours * 60 + minutes
        return int(total_minutes)
    except Exception:
        return None


def _compute_offer_metrics(offer_or_normalized: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comparable metrics for ranking:
    - price_amount: float (lower is better)
    - total_duration_minutes: int (lower is better)
    - total_connections: int (lower is better)

    Accepts either a raw Amadeus offer or a normalized entry that contains `raw`.
    """
    # Prefer computing from the normalized structure when available, per tool contract
    # Fall back to raw Amadeus offer structure otherwise.
    offer = offer_or_normalized.get("raw") or offer_or_normalized

    # Price
    price_amount: Optional[float] = None
    try:
        price_amount = float(
            (offer.get("price") or {}).get("grandTotal")
            or (offer.get("price") or {}).get("total")
        )
    except Exception:
        try:
            # Fallback to normalized min_amount
            if offer_or_normalized.get("min_amount") is not None:
                price_amount = float(offer_or_normalized.get("min_amount"))
        except Exception:
            price_amount = None

    # Duration and connections
    total_duration_minutes: Optional[int] = None
    total_connections = 0

    # 1) If normalized fields exist, compute based on them
    if isinstance(offer_or_normalized.get("segments"), list):
        try:
            # Per user requirement: number of connections obtained from len(segments)
            total_connections = len(offer_or_normalized.get("segments") or [])
        except Exception:
            total_connections = 0

        dep_s = offer_or_normalized.get("scheduled_departure")
        arr_s = offer_or_normalized.get("scheduled_arrival")
        try:
            if dep_s and arr_s:
                dep_dt = _to_datetime(dep_s)
                arr_dt = _to_datetime(arr_s)
                total_duration_minutes = max(
                    0, int((arr_dt - dep_dt).total_seconds() // 60)
                )
        except Exception:
            total_duration_minutes = None

    # 2) Otherwise, fall back to raw itineraries-based computation
    if total_duration_minutes is None and not total_connections:
        itineraries = offer.get("itineraries") or []
        if itineraries:
            accum_minutes = 0
            stops = 0
            for it in itineraries:
                # Duration
                dur = _parse_iso8601_duration_to_minutes(it.get("duration"))
                if dur is not None:
                    accum_minutes += dur
                else:
                    # Fallback: compute from first departure to last arrival if timestamps exist
                    segs = it.get("segments") or []
                    if segs:
                        try:
                            dep = (segs[0].get("departure") or {}).get("at")
                            arr = (segs[-1].get("arrival") or {}).get("at")
                            if dep and arr:
                                dep_dt = _to_datetime(dep)
                                arr_dt = _to_datetime(arr)
                                accum_minutes += max(
                                    0, int((arr_dt - dep_dt).total_seconds() // 60)
                                )
                        except Exception:
                            # ignore
                            pass

                # Connections (raw semantics: segments - 1 per itinerary)
                segs = it.get("segments") or []
                stops += max(0, len(segs) - 1)

            total_duration_minutes = accum_minutes
            total_connections = stops

    metrics = {
        "price_amount": price_amount,
        "total_duration_minutes": total_duration_minutes,
        "total_connections": total_connections,
    }
    return metrics


def rank_flights(
    filters_applied: Dict[str, Any],
    flights: List[Dict[str, Any]],
    top_k: int,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Score and rank flights. Higher score is better. Returns a list of entries:
    { "flight": <normalized_flight>, "score": float, "metrics": {...} }

    Factors: price_amount, total_duration_minutes, total_connections.
    Normalization: min-max across provided `flights` then inverted (lower is better).
    """
    if not flights or top_k <= 0:
        return []

    # Default weights
    weights = weights or {"price": 0.7, "duration": 0.2, "stops": 0.1}
    # Normalize weights to sum to 1
    total_w = float(sum(max(0.0, float(v)) for v in weights.values()) or 1.0)
    w_price = max(0.0, float(weights.get("price", 0.7))) / total_w
    w_dur = max(0.0, float(weights.get("duration", 0.2))) / total_w
    w_stops = max(0.0, float(weights.get("stops", 0.1))) / total_w

    # Compute metrics for each flight
    enriched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
        (flight, _compute_offer_metrics(flight)) for flight in flights
    ]

    # Gather ranges for min-max normalization
    def _min_max(
        values: List[Optional[float]],
    ) -> Tuple[Optional[float], Optional[float]]:
        vals = [v for v in values if v is not None]
        if not vals:
            return None, None
        return min(vals), max(vals)

    p_min, p_max = _min_max([m.get("price_amount") for _, m in enriched])
    d_min, d_max = _min_max([m.get("total_duration_minutes") for _, m in enriched])
    s_min, s_max = _min_max([float(m.get("total_connections")) for _, m in enriched])

    def _inv_norm(
        val: Optional[float], lo: Optional[float], hi: Optional[float]
    ) -> float:
        """Invert min-max normalize to 0..1 where lower input => higher score.

        If any value is missing, returns 0. If the range collapses (hi <= lo),
        treat all values as equally good and return 1.
        """
        if val is None or lo is None or hi is None:
            return 0.0
        lo_f = float(lo)
        hi_f = float(hi)
        v = float(val)
        if hi_f <= lo_f:
            return 1.0
        # Equivalent to 1 - (v - lo) / (hi - lo) but slightly clearer
        score = (hi_f - v) / (hi_f - lo_f)
        return max(0.0, min(1.0, score))

    ranked: List[Dict[str, Any]] = []
    for flight, m in enriched:
        score = (
            w_price * _inv_norm(m.get("price_amount"), p_min, p_max)
            + w_dur * _inv_norm(m.get("total_duration_minutes"), d_min, d_max)
            + w_stops * _inv_norm(float(m.get("total_connections", 0)), s_min, s_max)
        )
        ranked.append({"flight": flight, "score": float(score), "metrics": m})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[: int(top_k)]


# ----------------------
# Public convenience helpers to call from tools or chains
# ----------------------
def get_amadeus_client() -> AmadeusClient:
    env = _env("AMADEUS_ENV", "test") or "test"
    base_url = _env("AMADEUS_BASE_URL")
    return AmadeusClient(environment=env, base_url=base_url)


if __name__ == "__main__":
    # demo_price_flight_offers()
    # if os.getenv("RUN_PRICING_DEMO") == "1":
    #     demo_price_flight_offers()
    # else:
    #     client = get_amadeus_client()
    #     print(client._auth_headers())
    client = get_amadeus_client()
    print(client._auth_headers())