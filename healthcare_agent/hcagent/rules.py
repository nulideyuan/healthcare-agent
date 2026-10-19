"""Deterministic clinical and operational rules. Thresholds live here, not in the LLM.

Clinical scoring follows NEWS2 component bands (RCP 2017). Consciousness level and
supplemental-oxygen status are not in the dataset, so the score is labelled partial.
Synthetic research prototype: not for patient care.
"""

from __future__ import annotations

from typing import Any

PLAUSIBLE_RANGES = {
    "heart_rate_bpm": (20, 250),
    "respiratory_rate_bpm": (3, 70),
    "spo2_pct": (50, 100),
    "temperature_c": (30.0, 43.5),
    "systolic_bp_mmHg": (50, 260),
    "diastolic_bp_mmHg": (20, 160),
}

VITAL_LABELS = {
    "heart_rate_bpm": ("heart rate", "bpm"),
    "respiratory_rate_bpm": ("respiratory rate", "/min"),
    "spo2_pct": ("SpO2", "%"),
    "temperature_c": ("temperature", "°C"),
    "systolic_bp_mmHg": ("systolic BP", "mmHg"),
}

# Operational thresholds
ED_OVER_CAPACITY = 1.00
ED_SEVERE = 0.95
ED_BUSY = 0.85
FOUR_HOUR_MINUTES = 240
WARD_NEAR_CAPACITY = 0.95
WARD_MIN_AVAILABLE = 1
STALE_SNAPSHOT_MINUTES = 120
STALE_VITALS_HOURS = 24

HYPERCAPNIC_CONDITIONS = {"COPD"}


def _band(value: float, bands: list[tuple[float, int]], top: int) -> int:
    """bands: ascending (upper_bound_inclusive, score); values above the last bound score `top`."""
    for upper, score in bands:
        if value <= upper:
            return score
    return top


def news2_components(obs: dict[str, Any], spo2_scale_2: bool) -> dict[str, int]:
    rr = _band(obs["respiratory_rate_bpm"], [(8, 3), (11, 1), (20, 0), (24, 2)], 3)
    if spo2_scale_2:
        # Scale 2 assumes room air for >= 93% because oxygen therapy status is unknown.
        spo2 = _band(obs["spo2_pct"], [(83, 3), (85, 2), (87, 1)], 0)
    else:
        spo2 = _band(obs["spo2_pct"], [(91, 3), (93, 2), (95, 1)], 0)
    sbp = _band(obs["systolic_bp_mmHg"], [(90, 3), (100, 2), (110, 1), (219, 0)], 3)
    hr = _band(obs["heart_rate_bpm"], [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2)], 3)
    temp = _band(obs["temperature_c"], [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1)], 2)
    return {
        "respiratory_rate_bpm": rr,
        "spo2_pct": spo2,
        "systolic_bp_mmHg": sbp,
        "heart_rate_bpm": hr,
        "temperature_c": temp,
    }


def news2_partial(obs: dict[str, Any], conditions: list[str]) -> dict[str, Any]:
    scale_2 = bool(HYPERCAPNIC_CONDITIONS & set(conditions))
    comps = news2_components(obs, scale_2)
    total = sum(comps.values())
    if total >= 7:
        band = "high"
    elif total >= 5:
        band = "medium"
    elif max(comps.values()) == 3:
        band = "low-medium"
    else:
        band = "low"
    return {
        "total": total,
        "band": band,
        "components": comps,
        "spo2_scale": "scale 2 (COPD target 88-92%)" if scale_2 else "scale 1",
        "excluded": ["consciousness (ACVPU)", "supplemental oxygen"],
        "observation_time": obs["timestamp"],
    }


def ed_crowding_level(occupancy: float) -> tuple[str, str]:
    if occupancy >= ED_OVER_CAPACITY:
        return "over capacity", "critical"
    if occupancy >= ED_SEVERE:
        return "severely crowded", "critical"
    if occupancy >= ED_BUSY:
        return "busy", "warning"
    return "normal", "info"


def ward_pressure(ward: dict[str, Any]) -> str | None:
    if ward["available_beds"] <= 0:
        return "full"
    if ward["occupancy_ratio"] >= WARD_NEAR_CAPACITY or ward["available_beds"] <= WARD_MIN_AVAILABLE:
        return "near capacity"
    return None
