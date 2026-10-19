"""Diabetes-cohort tools (ShanghaiT2DM) for the two specialist agents.

Glucose Prediction Agent    dm_get_glucose_trend, dm_forecast_glucose, dm_get_forecast_accuracy
Patient Context Agent       dm_get_patient_profile, dm_get_medication_events, dm_get_meal_responses,
                            dm_find_similar_patterns, recall_session_memory

All of them are deterministic. Forecast numbers come from the trained model (dm_model.py), never
from the LLM. Every time-dependent tool takes `as_of` (the replay clock) and sees no data after it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .dm_data import DiabetesRepository, minutes_between, shift
from .dm_model import GlucoseForecaster
from .schemas import Role, ToolResult
from .tools import TS_PATTERN, ToolSpec, _error, _result

DM_ID = r"^DM\d{4}$"
CLINICAL_ROLES = frozenset({Role.PATIENT, Role.CLINICIAN})
ERRORS = ("INVALID_INPUT", "NOT_FOUND", "NO_DATA", "ACCESS_DENIED")
PREDICTION_AGENT, CONTEXT_AGENT = "glucose_prediction", "patient_context"
STALE_READING_MIN = 30


@dataclass
class DiabetesServices:
    repo: DiabetesRepository
    forecaster: GlucoseForecaster


# --------------------------------------------------------------------------- input schemas


class DMArgs(BaseModel):
    patient_id: str = Field(pattern=DM_ID, description="Diabetes-cohort patient ID, e.g. DM2045")


class DMTimedArgs(DMArgs):
    as_of: str | None = Field(default=None, pattern=TS_PATTERN,
                              description="Replay time YYYY-MM-DDTHH:MM. Omit for the end of the latest recording.")


class TrendArgs(DMTimedArgs):
    hours: int = Field(default=3, ge=1, le=24, description="Hours of CGM history to return")


class MedicationArgs(DMTimedArgs):
    hours: int = Field(default=72, ge=1, le=336, description="Look-back window in hours")


class MealArgs(DMTimedArgs):
    limit: int = Field(default=5, ge=1, le=10, description="Number of most recent meals")


class PatternArgs(DMTimedArgs):
    top_k: int = Field(default=3, ge=1, le=5, description="Number of similar past episodes")


class RecallArgs(BaseModel):
    patient_id: str = Field(pattern=r"^(P|DM)\d{4}$", description="Patient whose retrieved facts to list")


# --------------------------------------------------------------------------- helpers


def _resolve(svc: DiabetesServices, tool: str, a: DMArgs) -> tuple[str | None, ToolResult | None]:
    if not svc.repo.patient_exists(a.patient_id):
        return None, _error(tool, a.model_dump(), "dm_patients", "NOT_FOUND", f"No diabetes-cohort patient {a.patient_id}")
    as_of = getattr(a, "as_of", None) or svc.repo.default_as_of(a.patient_id)
    return as_of, None


def _range_stats(values: list[float]) -> dict[str, Any]:
    g = np.asarray(values, dtype=float)
    pct = lambda m: round(float(m.mean() * 100), 1)  # noqa: E731
    return {"n": int(g.size), "mean_mgdl": round(float(g.mean()), 1), "min_mgdl": float(g.min()), "max_mgdl": float(g.max()),
            "tbr_lt54_pct": pct(g < 54), "tbr_lt70_pct": pct(g < 70), "tir_70_180_pct": pct((g >= 70) & (g <= 180)),
            "tar_gt180_pct": pct(g > 180), "tar_gt250_pct": pct(g > 250)}


def _reading_near(rows: list[dict], ts: str, tol_min: float = 8) -> dict | None:
    best = min(rows, key=lambda r: abs(minutes_between(r["ts"], ts)), default=None)
    return best if best and abs(minutes_between(best["ts"], ts)) <= tol_min else None


# --------------------------------------------------------------------------- prediction agent tools


def dm_get_glucose_trend(svc: DiabetesServices, now: str, a: TrendArgs) -> ToolResult:
    name = "dm_get_glucose_trend"
    as_of, err = _resolve(svc, name, a)
    if err:
        return err
    rows = svc.repo.cgm(a.patient_id, shift(as_of, -a.hours * 60), as_of)
    if not rows:
        return _error(name, a.model_dump(), "dm_cgm", "NO_DATA", f"No CGM readings for {a.patient_id} in the {a.hours} h before {as_of}")
    g = [r["glucose_mgdl"] for r in rows]
    last = rows[-1]

    def rate(k: int) -> float | None:
        if len(rows) <= k or abs(minutes_between(rows[-1 - k]["ts"], last["ts"]) - 15 * k) > 1:
            return None
        return round((g[-1] - g[-1 - k]) / (15 * k), 2)

    day = svc.repo.cgm(a.patient_id, shift(as_of, -1440), as_of)
    data = {
        "patient_id": a.patient_id, "as_of": as_of, "recording_id": last["recording_id"],
        "current": {"ts": last["ts"], "glucose_mgdl": last["glucose_mgdl"]},
        "minutes_since_last_reading": round(minutes_between(last["ts"], as_of)),
        "rate_15min_mgdl_per_min": rate(1), "rate_30min_mgdl_per_min": rate(2),
        "window_hours": a.hours, "readings": [{"ts": r["ts"], "glucose_mgdl": r["glucose_mgdl"]} for r in rows],
        "last_24h": _range_stats([r["glucose_mgdl"] for r in day]),
    }
    return _result(name, a, "dm_cgm", data, last["ts"])


def dm_forecast_glucose(svc: DiabetesServices, now: str, a: DMTimedArgs) -> ToolResult:
    name = "dm_forecast_glucose"
    as_of, err = _resolve(svc, name, a)
    if err:
        return err
    rec = svc.repo.recording_at(a.patient_id, as_of)
    last = svc.repo.last_reading(a.patient_id, as_of)
    if rec is None or last is None:
        return _error(name, a.model_dump(), "dm_forecast_model", "NO_DATA", f"No CGM data for {a.patient_id} before {as_of}")
    if minutes_between(last["ts"], as_of) > STALE_READING_MIN:
        return _error(name, a.model_dump(), "dm_forecast_model", "NO_DATA",
                      f"No CGM reading in the {STALE_READING_MIN} min before {as_of} (last reading {last['ts']}); cannot forecast")
    res = svc.forecaster.predict(a.patient_id, svc.repo.recording_data(rec, as_of))
    if res is None:
        return _error(name, a.model_dump(), "dm_forecast_model", "NO_DATA",
                      "Forecast needs 2 h of uninterrupted 15-min CGM history before the current reading")
    for f in res["forecasts"]:
        f["target_time"] = shift(last["ts"], f["horizon_min"])
    data = {"patient_id": a.patient_id, "as_of": as_of, "base_time": last["ts"], "recording_id": rec["recording_id"],
            "model": svc.forecaster.name, "model_version": svc.forecaster.version,
            "recording_quality_flag": rec["quality_flag"], **res}
    return _result(name, a, "dm_forecast_model", data, last["ts"])


def dm_get_forecast_accuracy(svc: DiabetesServices, now: str, a: DMArgs) -> ToolResult:
    name = "dm_get_forecast_accuracy"
    _, err = _resolve(svc, name, a)
    if err:
        return err
    data = svc.forecaster.accuracy(a.patient_id)
    data["patient_id"] = a.patient_id
    if data["this_patient"] is None:
        data["this_patient_note"] = "No validation results for this patient (recording excluded for data quality)."
    return _result(name, a, "dm_model_eval", data, None)


# --------------------------------------------------------------------------- context agent tools

LAB_FIELDS = ["hba1c_pct", "hba1c_mmol_mol", "fasting_glucose_mgdl", "postprandial_2h_glucose_mgdl",
              "fasting_c_peptide_nmol_l", "postprandial_2h_c_peptide_nmol_l", "fasting_insulin_pmol_l",
              "postprandial_2h_insulin_pmol_l", "glycated_albumin_pct", "total_cholesterol_mmol_l",
              "triglyceride_mmol_l", "hdl_mmol_l", "ldl_mmol_l", "creatinine_umol_l", "egfr_ml_min",
              "uric_acid", "bun_mmol_l"]


def dm_get_patient_profile(svc: DiabetesServices, now: str, a: DMArgs) -> ToolResult:
    name = "dm_get_patient_profile"
    _, err = _resolve(svc, name, a)
    if err:
        return err
    recs = svc.repo.recordings(a.patient_id)
    r = recs[-1]
    labs = {k: r[k] for k in LAB_FIELDS}
    data = {
        "patient_id": a.patient_id,
        "demographics": {k: r[k] for k in ("sex", "age", "height_m", "weight_kg", "bmi", "smoking_pack_years", "alcohol")},
        "diabetes": {k: r[k] for k in ("diabetes_duration_years", "acute_complications", "macrovascular_complications",
                                       "microvascular_complications", "comorbidities")},
        "labs": labs,
        "labs_not_recorded": [k for k, v in labs.items() if v is None],
        "labs_from_recording": r["recording_id"],
        "prescribed_agents": {"hypoglycemic_agents": r["hypoglycemic_agents"], "other_agents": r["other_agents"]},
        "hypoglycemia_history_field": r["hypoglycemia_history"],
        "hypoglycemia_history_note": "Summary-table field; it often disagrees with CGM-detected lows, so treat it as clinical history only.",
        "recordings": [{k: x[k] for k in ("recording_id", "cgm_start", "cgm_end", "n_cgm", "quality_flag")} for x in recs],
    }
    return _result(name, a, "dm_recordings", data, None)


def dm_get_medication_events(svc: DiabetesServices, now: str, a: MedicationArgs) -> ToolResult:
    name = "dm_get_medication_events"
    as_of, err = _resolve(svc, name, a)
    if err:
        return err
    rows = svc.repo.medications(a.patient_id, shift(as_of, -a.hours * 60), as_of)
    if not rows:
        return _error(name, a.model_dump(), "dm_medications", "NO_DATA",
                      f"No insulin or oral-agent events recorded for {a.patient_id} in the {a.hours} h before {as_of}")
    summary: dict[str, dict[str, Any]] = {}
    for m in rows:
        key = m["route"] + (f"/{m['insulin_type']}" if m["insulin_type"] and m["route"] == "insulin_sc" else "")
        s = summary.setdefault(key, {"events": 0, "total_dose": 0.0, "unit": m["unit"]})
        s["events"] += 1
        if m["dose"] is not None and m["route"] != "insulin_pump_basal":
            s["total_dose"] = round(s["total_dose"] + m["dose"], 2)
    data = {"patient_id": a.patient_id, "as_of": as_of, "window_hours": a.hours, "summary_by_route": summary,
            "events": [{k: m[k] for k in ("ts", "route", "insulin_type", "description", "dose", "unit")} for m in rows]}
    return _result(name, a, "dm_medications", data, rows[-1]["ts"])


def dm_get_meal_responses(svc: DiabetesServices, now: str, a: MealArgs) -> ToolResult:
    name = "dm_get_meal_responses"
    as_of, err = _resolve(svc, name, a)
    if err:
        return err
    meals = svc.repo.meals(a.patient_id, "0000", as_of)[-a.limit:][::-1]
    if not meals:
        return _error(name, a.model_dump(), "dm_meals", "NO_DATA", f"No meals recorded for {a.patient_id} before {as_of}")
    out = []
    for m in meals:
        end = min(shift(m["ts"], 120), as_of)
        rows = svc.repo.cgm(a.patient_id, shift(m["ts"], -16), end)
        pre = _reading_near([r for r in rows if r["ts"] <= m["ts"]], m["ts"], 15)
        post = [r for r in rows if r["ts"] > m["ts"]]
        insulin = svc.repo.medications(a.patient_id, shift(m["ts"], -30), shift(m["ts"], 30))
        item = {"meal_ts": m["ts"], "description": m["description_en"], "total_grams": m["total_grams"],
                "pre_meal_mgdl": pre["glucose_mgdl"] if pre else None,
                "window_complete": shift(m["ts"], 120) <= as_of,
                "insulin_within_30min": [i["description"] for i in insulin if i["route"] != "oral"],
                "oral_agents_within_30min": [i["description"] for i in insulin if i["route"] == "oral"]}
        if pre and post:
            peak = max(post, key=lambda r: r["glucose_mgdl"])
            at2h = _reading_near(post, shift(m["ts"], 120))
            item.update({"peak_mgdl": peak["glucose_mgdl"], "rise_mgdl": round(peak["glucose_mgdl"] - pre["glucose_mgdl"], 1),
                         "minutes_to_peak": round(minutes_between(m["ts"], peak["ts"])),
                         "glucose_2h_mgdl": at2h["glucose_mgdl"] if at2h else None,
                         "min_after_meal_mgdl": min(r["glucose_mgdl"] for r in post)})
        out.append(item)
    rises = [x["rise_mgdl"] for x in out if "rise_mgdl" in x]
    summary = {"meals": len(out), "with_glucose_response": len(rises),
               "median_rise_mgdl": round(statistics.median(rises), 1) if rises else None,
               "meals_peaking_above_180": sum(1 for x in out if x.get("peak_mgdl", 0) > 180),
               "meals_followed_by_lt70": sum(1 for x in out if x.get("min_after_meal_mgdl", 999) < 70)}
    return _result(name, a, "dm_meals", {"patient_id": a.patient_id, "as_of": as_of, "summary": summary, "meals": out},
                   meals[0]["ts"])


def dm_find_similar_patterns(svc: DiabetesServices, now: str, a: PatternArgs) -> ToolResult:
    """Nearest past 2-h CGM windows (by RMSE) and what happened in the following hour."""
    name = "dm_find_similar_patterns"
    as_of, err = _resolve(svc, name, a)
    if err:
        return err
    rows = svc.repo.cgm(a.patient_id, "0000", as_of)
    if len(rows) < 9:
        return _error(name, a.model_dump(), "dm_cgm", "NO_DATA", "Not enough CGM history")
    t = np.array([minutes_between(rows[0]["ts"], r["ts"]) for r in rows])
    g = np.array([r["glucose_mgdl"] for r in rows])
    q_idx = np.arange(len(g) - 9, len(g))
    if not np.allclose(np.diff(t[q_idx]), 15) or minutes_between(rows[-1]["ts"], as_of) > STALE_READING_MIN:
        return _error(name, a.model_dump(), "dm_cgm", "NO_DATA", "The last 2 h before as_of are not a regular CGM series")
    query, q_start = g[q_idx], t[q_idx[0]]

    cands = []
    for e in range(8, len(g) - 4):                       # window [e-8, e], outcome e+1..e+4
        s = e - 8
        if t[e + 4] > q_start or not np.allclose(np.diff(t[s:e + 5]), 15):
            continue
        cands.append((float(np.sqrt(np.mean((g[s:e + 1] - query) ** 2))), s, e))
    if not cands:
        return _error(name, a.model_dump(), "dm_cgm", "NO_DATA",
                      "No earlier CGM history with a complete following hour to compare against")
    chosen: list[tuple[float, int, int]] = []
    for c in sorted(cands):
        if all(abs(t[c[2]] - t[x[2]]) >= 120 for x in chosen):
            chosen.append(c)
        if len(chosen) == a.top_k:
            break

    matches = []
    for dist, s, e in chosen:
        start, end = rows[s]["ts"], rows[e]["ts"]
        nxt = g[e + 1:e + 5]
        meds = svc.repo.medications(a.patient_id, shift(start, -60), end)
        matches.append({
            "window_start": start, "window_end": end, "rmse_to_current_mgdl": round(dist, 1),
            "glucose_at_end_mgdl": float(g[e]), "glucose_plus_30_mgdl": float(nxt[1]), "glucose_plus_60_mgdl": float(nxt[3]),
            "min_next_60_mgdl": float(nxt.min()), "fell_below_70_next_60": bool((nxt < 70).any()),
            "meal_in_window": bool(svc.repo.meals(a.patient_id, shift(start, -60), end)),
            "insulin_in_window": [m["description"] for m in meds if m["route"] != "oral"][:3],
        })
    data = {"patient_id": a.patient_id, "as_of": as_of,
            "current_window": {"start": rows[q_idx[0]]["ts"], "end": rows[-1]["ts"], "glucose_mgdl": query.tolist()},
            "history_searched": {"from": rows[0]["ts"], "windows_compared": len(cands)},
            "matches": matches,
            "summary": {"matches": len(matches),
                        "followed_by_lt70": sum(m["fell_below_70_next_60"] for m in matches),
                        "median_change_60min_mgdl": round(statistics.median(m["glucose_plus_60_mgdl"] - m["glucose_at_end_mgdl"]
                                                                            for m in matches), 1)}}
    return _result(name, a, "dm_cgm_history", data, rows[-1]["ts"])


def recall_session_memory(mem, now: str, a: RecallArgs) -> ToolResult:
    """What has already been retrieved about the patient: this session's tree and this user's earlier sessions."""
    name = "recall_session_memory"
    branch = mem.tree.patients.get(a.patient_id) if mem is not None and mem.tree is not None else None
    earlier = mem.earlier_turns(a.patient_id) if mem is not None else []
    if (branch is None or not (branch.facts or branch.turns)) and not earlier:
        return _error(name, a.model_dump(), "session_memory", "NO_DATA",
                      f"Nothing has been retrieved about {a.patient_id} in this or earlier conversations.")
    data = {"patient_id": a.patient_id,
            "this_session": {
                "facts": [{"tool": f.tool, "source": f.result.source, "data_timestamp": f.result.data_timestamp,
                           "retrieved_in": f.request_id, "args": f.args} for f in (branch.facts.values() if branch else [])],
                # Earlier answers are included so any value restated from them is checkable evidence.
                "earlier_questions": [{"turn": x.turn_no, "question": x.question, "route": x.route,
                                       "answer_excerpt": x.answer[:400]} for x in (branch.turns if branch else [])]},
            "earlier_sessions": earlier}
    return _result(name, a, "session_memory", data, None)


# --------------------------------------------------------------------------- registry

DM_TOOLS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("dm_get_glucose_trend",
             "Current CGM glucose, rate of change, the readings of the last few hours and 24-h time-in-range for a diabetes-cohort patient.",
             TrendArgs, "current reading, rates (mg/dL/min), readings[], last_24h range stats",
             "patient", CLINICAL_ROLES, ERRORS, "dm_cgm", dm_get_glucose_trend, "diabetes", PREDICTION_AGENT),
    ToolSpec("dm_forecast_glucose",
             "Run the validated time-series model to forecast glucose 30 and 60 min ahead, with 90% intervals and the main contributing inputs.",
             DMTimedArgs, "forecasts[]: horizon_min, predicted_mgdl, lower_90_mgdl, upper_90_mgdl, factor_effects_mgdl",
             "patient", CLINICAL_ROLES, ERRORS, "dm_forecast_model", dm_forecast_glucose, "diabetes", PREDICTION_AGENT),
    ToolSpec("dm_get_forecast_accuracy",
             "Validation accuracy of the glucose forecaster: overall, for this patient, and for the hypo/hyper risk rules.",
             DMArgs, "out-of-fold RMSE, MAE, MARD, Clarke zones, risk-rule sensitivity/precision",
             "patient", CLINICAL_ROLES, ERRORS, "dm_model_eval", dm_get_forecast_accuracy, "diabetes", PREDICTION_AGENT),
    ToolSpec("dm_get_patient_profile",
             "Diabetes-cohort patient profile: demographics, diabetes history, complications, labs (HbA1c, C-peptide, lipids, renal), prescribed agents, recordings.",
             DMArgs, "demographics, diabetes, labs, labs_not_recorded, prescribed_agents, recordings[]",
             "patient", CLINICAL_ROLES, ERRORS, "dm_recordings", dm_get_patient_profile, "diabetes", CONTEXT_AGENT),
    ToolSpec("dm_get_medication_events",
             "Recorded insulin (s.c., pump bolus/basal, i.v.) and oral hypoglycaemic agent events in a time window.",
             MedicationArgs, "events[]: ts, route, insulin_type, description, dose; summary_by_route",
             "patient", CLINICAL_ROLES, ERRORS, "dm_medications", dm_get_medication_events, "diabetes", CONTEXT_AGENT),
    ToolSpec("dm_get_meal_responses",
             "The patient's most recent meals and the CGM response after each (pre-meal, peak, rise, 2-h value).",
             MealArgs, "meals[]: meal_ts, description, pre_meal_mgdl, peak_mgdl, rise_mgdl, glucose_2h_mgdl; summary",
             "patient", CLINICAL_ROLES, ERRORS, "dm_meals", dm_get_meal_responses, "diabetes", CONTEXT_AGENT),
    ToolSpec("dm_find_similar_patterns",
             "Find earlier 2-h CGM windows of this patient most similar to the current one, and what happened in the hour after each.",
             PatternArgs, "matches[]: window, rmse_to_current_mgdl, glucose_plus_30/60_mgdl, fell_below_70_next_60; summary",
             "patient", CLINICAL_ROLES, ERRORS, "dm_cgm_history", dm_find_similar_patterns, "diabetes", CONTEXT_AGENT),
    ToolSpec("recall_session_memory",
             "List what has already been retrieved about a patient: this conversation's memory tree and the same user's earlier sessions.",
             RecallArgs, "this_session{facts[], earlier_questions[]}, earlier_sessions[] (question, asked_at, answer_excerpt)",
             "patient", CLINICAL_ROLES, ("NO_DATA", "ACCESS_DENIED"), "session_memory", recall_session_memory, "memory", CONTEXT_AGENT),
]}
