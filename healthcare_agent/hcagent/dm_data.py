"""Read-only access to the ShanghaiT2DM database (built by diabetes/etl.py).

Each diabetes patient has a replay clock, `as_of`. It defaults to the last CGM reading of the
patient's latest recording, and every query only returns data at or before it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .dm_features import RecordingData

TS = "%Y-%m-%dT%H:%M"


def shift(ts: str, minutes: float) -> str:
    return (datetime.fromisoformat(ts) + timedelta(minutes=minutes)).strftime(TS)


def minutes_between(a: str, b: str) -> float:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 60


class DiabetesRepository:
    def __init__(self, db_path: Path):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Diabetes database not found: {db_path}. Run: python3 diabetes/etl.py")
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def _all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        r = self._conn.execute(sql, params).fetchone()
        return dict(r) if r else None

    # ------------------------------------------------------------------ patients / recordings

    def patient_ids(self) -> list[str]:
        return [r["patient_id"] for r in self._all("SELECT patient_id FROM dm_patients ORDER BY patient_id")]

    def patient_exists(self, patient_id: str) -> bool:
        return self._one("SELECT 1 AS x FROM dm_patients WHERE patient_id=?", (patient_id,)) is not None

    def recordings(self, patient_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM dm_recordings WHERE patient_id=? ORDER BY cgm_start", (patient_id,))

    def default_as_of(self, patient_id: str) -> str | None:
        r = self._one("SELECT MAX(cgm_end) AS t FROM dm_recordings WHERE patient_id=?", (patient_id,))
        return r["t"] if r else None

    def recording_at(self, patient_id: str, as_of: str) -> dict[str, Any] | None:
        """The recording covering `as_of`, else the latest one that started before it."""
        return self._one(
            "SELECT * FROM dm_recordings WHERE patient_id=? AND cgm_start<=? ORDER BY cgm_start DESC LIMIT 1",
            (patient_id, as_of))

    # ------------------------------------------------------------------ time series

    def cgm(self, patient_id: str, start: str, end: str) -> list[dict[str, Any]]:
        return self._all("SELECT recording_id, ts, glucose_mgdl FROM dm_cgm WHERE patient_id=? AND ts>? AND ts<=? ORDER BY ts",
                         (patient_id, start, end))

    def last_reading(self, patient_id: str, as_of: str) -> dict[str, Any] | None:
        return self._one("SELECT recording_id, ts, glucose_mgdl FROM dm_cgm WHERE patient_id=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                         (patient_id, as_of))

    def meals(self, patient_id: str, start: str, end: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM dm_meals WHERE patient_id=? AND ts>? AND ts<=? ORDER BY ts", (patient_id, start, end))

    def medications(self, patient_id: str, start: str, end: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM dm_medications WHERE patient_id=? AND ts>? AND ts<=? ORDER BY ts", (patient_id, start, end))

    def recording_data(self, recording: dict[str, Any], until: str) -> RecordingData:
        """Everything the forecaster needs for one recording, truncated at `until` (no look-ahead)."""
        rid = recording["recording_id"]
        cgm = self._all("SELECT ts, glucose_mgdl FROM dm_cgm WHERE recording_id=? AND ts<=? ORDER BY ts", (rid, until))
        meals = self._all("SELECT ts, total_grams FROM dm_meals WHERE recording_id=? AND ts<=? ORDER BY ts", (rid, until))
        meds = self._all("SELECT ts, route, insulin_type, dose FROM dm_medications WHERE recording_id=? AND ts<=? ORDER BY ts",
                         (rid, until))
        static = {k: recording[k] for k in ("age", "bmi", "hba1c_pct", "diabetes_duration_years", "sex")}
        return RecordingData([c["ts"] for c in cgm], [c["glucose_mgdl"] for c in cgm],
                             [(m["ts"], m["total_grams"]) for m in meals],
                             [(m["ts"], m["route"], m["insulin_type"], m["dose"]) for m in meds], static)
