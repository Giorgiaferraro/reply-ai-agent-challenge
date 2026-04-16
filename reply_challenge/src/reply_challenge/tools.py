from __future__ import annotations

import importlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def detect_transaction_id(transaction: dict[str, Any]) -> str:
    for key in ("id", "transaction_id", "tx_id", "uuid"):
        if transaction.get(key):
            return str(transaction[key])
    return ""


def detect_user_id(transaction: dict[str, Any]) -> str:
    for key in ("user_id", "customer_id", "account_id", "owner_id"):
        if transaction.get(key):
            return str(transaction[key])
    return "unknown-user"


def detect_amount(transaction: dict[str, Any]) -> float:
    for key in ("amount", "value", "transaction_amount"):
        if key in transaction:
            return _safe_float(transaction.get(key), default=0.0)
    return 0.0


def detect_sender(transaction: dict[str, Any]) -> str:
    for key in ("sender", "sender_iban", "from", "source", "counterparty"):
        if transaction.get(key):
            return str(transaction[key])
    return ""


def detect_recipient(transaction: dict[str, Any]) -> str:
    for key in ("recipient", "recipient_iban", "to", "destination", "merchant"):
        if transaction.get(key):
            return str(transaction[key])
    return ""


def detect_timestamp(transaction: dict[str, Any]) -> datetime | None:
    for key in ("timestamp", "created_at", "datetime", "time"):
        value = _parse_iso_datetime(transaction.get(key))
        if value:
            return value
    return None


def detect_text_payload(transaction: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("description", "message", "sms", "email", "note", "text"):
        value = transaction.get(key)
        if value:
            chunks.append(str(value))
    return "\n".join(chunks).strip()


def detect_audio_path(transaction: dict[str, Any], dataset_root: Path) -> Path | None:
    for key in ("audio", "audio_file", "voicemail", "voice_message"):
        value = transaction.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return path
        candidate = (dataset_root / path).resolve()
        if candidate.exists():
            return candidate
    return None


def detect_coordinates(transaction: dict[str, Any]) -> tuple[float, float] | None:
    if "location" in transaction and isinstance(transaction["location"], dict):
        location = transaction["location"]
        lat = location.get("lat", location.get("latitude"))
        lon = location.get("lon", location.get("lng", location.get("longitude")))
        if lat is not None and lon is not None:
            return (_safe_float(lat), _safe_float(lon))

    lat = transaction.get("lat", transaction.get("latitude"))
    lon = transaction.get("lon", transaction.get("lng", transaction.get("longitude")))
    if lat is not None and lon is not None:
        return (_safe_float(lat), _safe_float(lon))
    return None


def haversine_km(coord_a: tuple[float, float], coord_b: tuple[float, float]) -> float:
    lat1, lon1 = coord_a
    lat2, lon2 = coord_b
    radius_km = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


@dataclass
class LocationEvent:
    timestamp: datetime
    lat: float
    lon: float


class ForensicSharedMemory:
    """Process-wide singleton memory shared across all dataset analyses."""

    _instance: "ForensicSharedMemory | None" = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._lock = Lock()
        self.compromised_users: set[str] = set()
        self.suspicious_ips: set[str] = set()
        self.suspicious_ibans: set[str] = set()
        self.suspicious_ids: set[str] = set()
        self.last_location_by_user: dict[str, LocationEvent] = {}

    @classmethod
    def instance(cls) -> "ForensicSharedMemory":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def mark_user_compromised(self, user_id: str) -> None:
        if not user_id:
            return
        with self._lock:
            self.compromised_users.add(user_id)

    def is_user_compromised(self, user_id: str) -> bool:
        if not user_id:
            return False
        with self._lock:
            return user_id in self.compromised_users

    def mark_indicator(self, kind: str, value: str) -> None:
        if not value:
            return
        normalized = value.strip().lower()
        with self._lock:
            if kind == "ip":
                self.suspicious_ips.add(normalized)
            elif kind == "iban":
                self.suspicious_ibans.add(normalized)
            else:
                self.suspicious_ids.add(normalized)

    def is_indicator_suspicious(self, value: str) -> bool:
        normalized = (value or "").strip().lower()
        if not normalized:
            return False
        with self._lock:
            return (
                normalized in self.suspicious_ips
                or normalized in self.suspicious_ibans
                or normalized in self.suspicious_ids
            )

    def evaluate_impossible_travel(
        self,
        user_id: str,
        timestamp: datetime | None,
        coords: tuple[float, float] | None,
        speed_kmh_threshold: float = 950.0,
    ) -> dict[str, Any]:
        if not user_id or not timestamp or not coords:
            return {"flag": False, "distance_km": 0.0, "speed_kmh": 0.0}

        with self._lock:
            previous = self.last_location_by_user.get(user_id)
            self.last_location_by_user[user_id] = LocationEvent(timestamp, coords[0], coords[1])

        if not previous:
            return {"flag": False, "distance_km": 0.0, "speed_kmh": 0.0}

        delta_seconds = abs((timestamp - previous.timestamp).total_seconds())
        if delta_seconds <= 0:
            return {"flag": False, "distance_km": 0.0, "speed_kmh": 0.0}

        distance_km = haversine_km((previous.lat, previous.lon), coords)
        speed_kmh = distance_km / (delta_seconds / 3600.0)
        return {
            "flag": speed_kmh > speed_kmh_threshold,
            "distance_km": round(distance_km, 2),
            "speed_kmh": round(speed_kmh, 2),
        }


class RecurringPatternTracker:
    """Tracks recurring salary/rent patterns to reduce false positives."""

    def __init__(self) -> None:
        self.salary_sender_counter: dict[str, Counter[str]] = defaultdict(Counter)
        self.rent_recipient_counter: dict[str, Counter[str]] = defaultdict(Counter)

    @staticmethod
    def _is_salary_amount(amount: float) -> bool:
        return 2500.0 <= amount <= 3200.0

    @staticmethod
    def _is_rent_amount(amount: float) -> bool:
        return 700.0 <= amount <= 1200.0

    def update(self, transaction: dict[str, Any]) -> None:
        user_id = detect_user_id(transaction)
        amount = detect_amount(transaction)
        sender = detect_sender(transaction)
        recipient = detect_recipient(transaction)

        if self._is_salary_amount(amount) and sender:
            self.salary_sender_counter[user_id][sender.lower()] += 1
        if self._is_rent_amount(amount) and recipient:
            self.rent_recipient_counter[user_id][recipient.lower()] += 1

    def evaluate(self, transaction: dict[str, Any]) -> dict[str, bool]:
        user_id = detect_user_id(transaction)
        amount = detect_amount(transaction)
        sender = detect_sender(transaction).lower()
        recipient = detect_recipient(transaction).lower()

        flags = {
            "salary_unknown_sender": False,
            "duplicate_rent_risk": False,
            "recurring_legit": False,
        }

        if self._is_salary_amount(amount):
            if sender and self.salary_sender_counter[user_id][sender] > 0:
                flags["recurring_legit"] = True
            elif sender:
                flags["salary_unknown_sender"] = True

        if self._is_rent_amount(amount):
            if recipient and self.rent_recipient_counter[user_id][recipient] >= 2:
                flags["duplicate_rent_risk"] = True
            elif recipient and self.rent_recipient_counter[user_id][recipient] >= 1:
                flags["recurring_legit"] = True

        return flags


def compute_tier(transaction: dict[str, Any], anomaly_flags: dict[str, bool]) -> int:
    amount = detect_amount(transaction)
    high_value_anomaly = anomaly_flags.get("salary_unknown_sender") or anomaly_flags.get("duplicate_rent_risk")
    severe_amount_anomaly = amount > 2000 and anomaly_flags.get("any_behavioral_anomaly", False)

    if (amount > 1000 and high_value_anomaly) or severe_amount_anomaly:
        return 1
    if amount < 100:
        return 3
    return 2


VISHING_PATTERNS = {
    "urgency": re.compile(r"urgent|immediately|subito|adesso|ora|asap", re.IGNORECASE),
    "otp_request": re.compile(r"otp|one[- ]?time|codice|verification code|2fa", re.IGNORECASE),
    "threat": re.compile(r"blocked|sospeso|chiuso|legal|police|penalty|multa|minaccia", re.IGNORECASE),
    "credential_request": re.compile(r"password|pin|iban|cvv|credential|accesso", re.IGNORECASE),
}


def analyze_vishing_text(text: str) -> dict[str, Any]:
    lowered = (text or "").strip()
    hits: dict[str, int] = {}
    for key, pattern in VISHING_PATTERNS.items():
        matches = pattern.findall(lowered)
        if matches:
            hits[key] = len(matches)
    score = min(1.0, sum(hits.values()) / 4.0)
    return {
        "score": round(score, 2),
        "signals": hits,
        "is_suspicious": score >= 0.5,
    }


@lru_cache(maxsize=1)
def _load_whisper_model() -> Any:
    whisper = importlib.import_module("whisper")
    model_name = os.getenv("WHISPER_MODEL", "base")
    return whisper.load_model(model_name)


def transcribe_audio(audio_path: Path | None) -> str:
    if audio_path is None or not audio_path.exists():
        return ""
    try:
        model = _load_whisper_model()
        result = model.transcribe(str(audio_path))
        return str(result.get("text", "")).strip()
    except Exception:
        return ""


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)