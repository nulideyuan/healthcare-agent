"""Glucose-forecast features, shared by training (diabetes/train_forecaster.py) and the agent runtime.

Pure numpy, so the same code runs in the analysis Python and the MAF venv. For each CGM reading at
time t, features use only data at or before t (no look-ahead):

  * the CGM history over the previous 2 h (15-min grid): level, deltas, slopes, variability
  * time of day
  * meals: grams eaten in the last 2 h / 4 h, minutes since the last meal
  * insulin: rapid/premixed IU in the last 4 h (s.c., pump bolus, i.v.), basal s.c. IU in the last
    24 h, current pump basal rate
  * oral agents taken in the last 6 h
  * static: age, BMI, HbA1c, diabetes duration, sex

Targets are the change in glucose 30 and 60 min ahead (2 and 4 readings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

STEP_MIN = 15
LAGS = 8                       # 2 h of history
HORIZONS_MIN = (30, 60)
STATIC_DEFAULTS = {"age": 63.0, "bmi": 23.6, "hba1c_pct": 8.3, "diabetes_duration_years": 7.0}  # cohort medians

FEATURES = (
    ["g0"] + [f"d{k}" for k in range(1, LAGS + 1)]
    + ["slope15", "slope30", "slope60", "accel", "mean2h", "sd2h", "min2h", "max2h", "tod_sin", "tod_cos",
       "meal_g_2h", "meal_g_4h", "min_since_meal", "meal_last_30",
       "rapid_iu_4h", "basal_sc_iu_24h", "pump_basal_rate", "oral_6h",
       "age", "bmi", "hba1c_pct", "diabetes_duration_years", "male"]
)


def to_minutes(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp() / 60.0


@dataclass
class RecordingData:
    """One monitoring period. Times are 'YYYY-MM-DDTHH:MM' strings, sorted ascending."""

    cgm_ts: list[str]
    cgm: list[float]
    meals: list[tuple[str, float | None]] = field(default_factory=list)            # (ts, grams)
    meds: list[tuple[str, str, str | None, float | None]] = field(default_factory=list)  # (ts, route, insulin_type, dose)
    static: dict[str, float | str | None] = field(default_factory=dict)


def _window_sum(ev_t: np.ndarray, ev_v: np.ndarray, t: np.ndarray, width: float) -> np.ndarray:
    """Sum of event values with time in (t - width, t]."""
    if ev_t.size == 0:
        return np.zeros_like(t)
    c = np.concatenate([[0.0], np.cumsum(ev_v)])
    return c[np.searchsorted(ev_t, t, side="right")] - c[np.searchsorted(ev_t, t - width, side="right")]


def _events(items, select) -> tuple[np.ndarray, np.ndarray]:
    pts = sorted((to_minutes(x[0]), v) for x in items if (v := select(x)) is not None)
    return np.array([p[0] for p in pts], dtype=float), np.array([p[1] for p in pts], dtype=float)


def build_features(rec: RecordingData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X [n, F], valid_mask [n], minutes [n]). Rows lacking a regular 2-h history are invalid."""
    t = np.array([to_minutes(s) for s in rec.cgm_ts], dtype=float)
    g = np.asarray(rec.cgm, dtype=float)
    n = len(g)
    lag = np.full((n, LAGS + 1), np.nan)
    valid = np.ones(n, dtype=bool)
    for k in range(LAGS + 1):
        lag[k:, k] = g[: n - k] if k else g
        if k:
            regular = np.zeros(n, dtype=bool)
            regular[k:] = np.isclose(t[k:] - t[: n - k], k * STEP_MIN)
            valid &= regular
    g0 = lag[:, 0]
    hist = lag[:, : LAGS + 1]

    tod = np.array([datetime.fromisoformat(s).hour * 60 + datetime.fromisoformat(s).minute for s in rec.cgm_ts], float)
    meal_t, meal_g = _events(rec.meals, lambda m: m[1] if m[1] is not None else 0.0)
    meal_any_t, meal_any_v = _events(rec.meals, lambda m: 1.0)
    rapid_t, rapid_v = _events(rec.meds, lambda m: m[3] if m[1] in ("insulin_sc", "insulin_pump_bolus", "insulin_iv")
                               and m[2] in ("rapid", "premixed") and m[3] is not None else None)
    basal_t, basal_v = _events(rec.meds, lambda m: m[3] if m[1] == "insulin_sc" and m[2] == "basal" and m[3] is not None else None)
    pump_t, pump_v = _events(rec.meds, lambda m: m[3] if m[1] == "insulin_pump_basal" else None)
    oral_t, oral_v = _events(rec.meds, lambda m: 1.0 if m[1] == "oral" else None)

    last_meal_idx = np.searchsorted(meal_any_t, t, side="right") - 1
    since_meal = np.where(last_meal_idx >= 0, t - meal_any_t[np.clip(last_meal_idx, 0, None)] if meal_any_t.size else 360, 360)
    since_meal = np.clip(since_meal, 0, 360)
    pump_idx = np.searchsorted(pump_t, t, side="right") - 1
    pump_rate = np.where(pump_idx >= 0, pump_v[np.clip(pump_idx, 0, None)] if pump_v.size else 0.0, 0.0)

    s = rec.static
    static = [float(s.get(k) if s.get(k) is not None else d) for k, d in STATIC_DEFAULTS.items()]
    male = 1.0 if s.get("sex") == "male" else 0.0

    with np.errstate(invalid="ignore"):
        cols = (
            [g0] + [g0 - lag[:, k] for k in range(1, LAGS + 1)]
            + [(g0 - lag[:, 1]) / 15, (g0 - lag[:, 2]) / 30, (g0 - lag[:, 4]) / 60, g0 - 2 * lag[:, 1] + lag[:, 2],
               np.nanmean(hist, axis=1), np.nanstd(hist, axis=1), np.nanmin(hist, axis=1), np.nanmax(hist, axis=1),
               np.sin(2 * np.pi * tod / 1440), np.cos(2 * np.pi * tod / 1440),
               _window_sum(meal_t, meal_g, t, 120), _window_sum(meal_t, meal_g, t, 240), since_meal,
               (since_meal <= 30).astype(float),
               _window_sum(rapid_t, rapid_v, t, 240), _window_sum(basal_t, basal_v, t, 1440), pump_rate,
               _window_sum(oral_t, oral_v, t, 360)]
            + [np.full(n, v) for v in static] + [np.full(n, male)]
        )
    X = np.column_stack(cols)
    valid &= ~np.isnan(X).any(axis=1)
    return X, valid, t


def targets(rec: RecordingData, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Glucose change at each horizon, and a mask where the future reading exists on the regular grid."""
    g = np.asarray(rec.cgm, dtype=float)
    n = len(g)
    Y = np.full((n, len(HORIZONS_MIN)), np.nan)
    for j, h in enumerate(HORIZONS_MIN):
        k = h // STEP_MIN
        ok = np.zeros(n, dtype=bool)
        ok[: n - k] = np.isclose(t[k:] - t[: n - k], h)
        Y[: n - k, j] = g[k:] - g[: n - k]
        Y[~ok, j] = np.nan
    return Y, ~np.isnan(Y).any(axis=1)


def clarke_zone(ref: float, pred: float) -> str:
    """Clarke error grid zone for one (reference, prediction) pair in mg/dL."""
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= 7 / 5 * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or (175 / 3 <= ref <= 70 and pred >= 6 / 5 * ref):
        return "D"
    return "B"
