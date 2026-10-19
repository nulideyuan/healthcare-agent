"""Read-only access to the synthetic clinical datastore.

Every query is time-consistent with the simulated clock: records stamped after "now" are
invisible, and encounter fields that would reveal the future (disposition, admission,
discharge) are masked while the patient is still in the ED.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TS_FMT = "%Y-%m-%dT%H:%M"


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def fmt_ts(dt: datetime) -> str:
    return dt.strftime(TS_FMT)


def minutes_between(start: str, end: str) -> int:
    return int((parse_ts(end) - parse_ts(start)).total_seconds() // 60)


class ClinicalRepository:
    def __init__(self, db_path: Path):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def _all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ reference data

    def latest_snapshot_time(self) -> str:
        return self._one("SELECT MAX(timestamp) AS ts FROM ed_status")["ts"]

    def wards(self) -> list[str]:
        return [r["ward"] for r in self._all("SELECT DISTINCT ward FROM beds ORDER BY ward")]

    def patient_exists(self, patient_id: str) -> bool:
        return self._one("SELECT 1 AS x FROM patients WHERE patient_id = ?", (patient_id,)) is not None

    # ------------------------------------------------------------------ patients

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        if row is None:
            return None
        conditions = row.get("chronic_conditions") or "None"
        row["chronic_conditions"] = [] if conditions == "None" else conditions.split("|")
        row["bmi"] = round(row["weight_kg"] / (row["height_m"] ** 2), 1)
        return row

    # ------------------------------------------------------------------ encounters

    def get_encounters(
        self, patient_id: str, now: str, limit: int = 5, since: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM encounters WHERE patient_id = ? AND arrival_time <= ?"
        params: list[Any] = [patient_id, now]
        if since:
            sql += " AND arrival_time >= ?"
            params.append(since)
        sql += " ORDER BY arrival_time DESC LIMIT ?"
        params.append(limit)
        return [self._mask_encounter(e, now) for e in self._all(sql, tuple(params))]

    @staticmethod
    def _mask_encounter(e: dict[str, Any], now: str) -> dict[str, Any]:
        admission = e.get("admission_time") or None
        discharge = e.get("discharge_time") or None
        ed_departure = admission or discharge
        e["admission_time"], e["discharge_time"] = admission, discharge

        if ed_departure is None or ed_departure > now:
            e["status_at_now"] = "IN_ED"
            e["disposition"] = None
            e["admission_time"] = None
            e["discharge_time"] = None
            e["ed_departure_time"] = None
            e["ed_minutes"] = minutes_between(e["arrival_time"], now)
        else:
            e["ed_departure_time"] = ed_departure
            e["ed_minutes"] = minutes_between(e["arrival_time"], ed_departure)
            if admission and (discharge is None or discharge > now):
                e["status_at_now"] = "ADMITTED_INPATIENT"
                e["discharge_time"] = None
            else:
                e["status_at_now"] = "COMPLETED"
        return e

    # ------------------------------------------------------------------ vitals

    def get_vitals(
        self,
        patient_id: str,
        now: str,
        hours: int | None = None,
        encounter_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM vitals WHERE patient_id = ? AND timestamp <= ?"
        params: list[Any] = [patient_id, now]
        if hours is not None:
            sql += " AND timestamp >= ?"
            params.append(fmt_ts(parse_ts(now) - timedelta(hours=hours)))
        if encounter_id:
            sql += " AND encounter_id = ?"
            params.append(encounter_id)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return self._all(sql, tuple(params))

    def latest_vitals_time(self, patient_id: str, now: str) -> str | None:
        row = self._one(
            "SELECT MAX(timestamp) AS ts FROM vitals WHERE patient_id = ? AND timestamp <= ?",
            (patient_id, now),
        )
        return row["ts"] if row else None

    # ------------------------------------------------------------------ operations

    def get_ed_status(self, at: str, trend_hours: int = 0) -> list[dict[str, Any]]:
        """Latest snapshot at or before `at`, followed by up to `trend_hours` earlier snapshots."""
        return self._all(
            "SELECT * FROM ed_status WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT ?",
            (at, trend_hours + 1),
        )

    def ed_percentile(self, column: str, value: float, at: str, days: int = 30) -> float | None:
        """Share of hourly snapshots in the previous `days` with a value <= `value` (0-100)."""
        if column not in {"ed_occupancy_ratio", "patients_waiting", "patients_on_trolleys", "average_wait_minutes"}:
            raise ValueError(column)
        start = fmt_ts(parse_ts(at) - timedelta(days=days))
        row = self._one(
            f"SELECT COUNT(*) AS n, SUM({column} <= ?) AS le FROM ed_status WHERE timestamp > ? AND timestamp <= ?",
            (value, start, at),
        )
        return round(100 * row["le"] / row["n"]) if row and row["n"] else None

    def get_beds(self, at: str, ward: str | None = None) -> list[dict[str, Any]]:
        snap = self._one("SELECT MAX(timestamp) AS ts FROM beds WHERE timestamp <= ?", (at,))
        if not snap or not snap["ts"]:
            return []
        sql = "SELECT * FROM beds WHERE timestamp = ?"
        params: list[Any] = [snap["ts"]]
        if ward:
            sql += " AND lower(ward) = lower(?)"
            params.append(ward)
        return self._all(sql + " ORDER BY ward", tuple(params))