"""Deterministic glucose rules for the diabetes cohort. Thresholds live here, not in the LLM.

Glucose bands: International Consensus on Time in Range (Battelino et al., Diabetes Care 2019).
Trend arrows: CGM rate-of-change convention (mg/dL/min).
Risk levels combine the forecast with its 90% interval; their validated sensitivity and precision
come from the model artifact and are quoted in the findings.
"""

from __future__ import annotations

from typing import Any

from .schemas import Finding, SharedState

PREDICTION, CONTEXT = "glucose_prediction", "patient_context"
HYPO, SEVERE_HYPO, HYPER, SEVERE_HYPER = 70, 54, 180, 250
STALE_CGM_MIN = 30
HBA1C_TARGET_PCT = 7.0


def glucose_band(g: float) -> tuple[str, str]:
    if g < SEVERE_HYPO:
        return "level 2 hypoglycaemia (<54 mg/dL)", "critical"
    if g < HYPO:
        return "level 1 hypoglycaemia (54-69 mg/dL)", "warning"
    if g <= HYPER:
        return "in target range (70-180 mg/dL)", "info"
    if g <= SEVERE_HYPER:
        return "level 1 hyperglycaemia (181-250 mg/dL)", "warning"
    return "level 2 hyperglycaemia (>250 mg/dL)", "critical"


def trend_arrow(rate: float) -> str:
    for limit, label in ((-3, "falling rapidly (↓↓)"), (-2, "falling (↓)"), (-1, "falling slowly (↘)")):
        if rate <= limit:
            return label
    for limit, label in ((3, "rising rapidly (↑↑)"), (2, "rising (↑)"), (1, "rising slowly (↗)")):
        if rate >= limit:
            return label
    return "stable (→)"


def risk_level(point: float, bound: float, below: bool, threshold: float) -> str:
    beyond = (lambda v: v < threshold) if below else (lambda v: v > threshold)
    if beyond(point):
        return "high"
    if beyond(bound):
        return "elevated"
    return "not elevated"


def _f(state: SharedState, owner: str, category: str, code: str, msg: str, severity: str = "info", **ev) -> None:
    state.findings.append(Finding(category=category, code=code, message=msg, severity=severity, evidence=ev, owner=owner))


def validate_dm(state: SharedState, risk_perf: dict[str, Any] | None) -> None:
    gc: dict[str, Any] = {}

    if trend := state.last_ok("dm_get_glucose_trend"):
        d = trend.data
        cur = d["current"]
        band, sev = glucose_band(cur["glucose_mgdl"])
        gc.update(current_mgdl=cur["glucose_mgdl"], current_ts=cur["ts"], band=band, as_of=d["as_of"])
        _f(state, PREDICTION, "clinical", "GLUCOSE_CURRENT",
           f"Current CGM glucose {cur['glucose_mgdl']} mg/dL at {cur['ts']}: {band}.", sev)
        rate = d["rate_30min_mgdl_per_min"] if d["rate_30min_mgdl_per_min"] is not None else d["rate_15min_mgdl_per_min"]
        if rate is not None:
            gc["trend"] = trend_arrow(rate)
            _f(state, PREDICTION, "clinical", "GLUCOSE_TREND",
               f"Trend {trend_arrow(rate)}: {rate:+.2f} mg/dL/min over the last 30 min.", rate=rate)
        if d["minutes_since_last_reading"] > STALE_CGM_MIN:
            _f(state, PREDICTION, "data_quality", "STALE_CGM",
               f"Last CGM reading is {d['minutes_since_last_reading']} min before {d['as_of']}; it may not reflect the current glucose.",
               "warning")
        s = d["last_24h"]
        gc["last_24h"] = s
        _f(state, PREDICTION, "clinical", "GLUCOSE_24H",
           f"Last 24 h ({s['n']} readings): mean {s['mean_mgdl']} mg/dL, time in range {s['tir_70_180_pct']}%, "
           f"below 70 {s['tbr_lt70_pct']}%, above 180 {s['tar_gt180_pct']}%.")

    if fc := state.last_ok("dm_forecast_glucose"):
        d = fc.data
        gc["forecasts"] = d["forecasts"]
        for f in d["forecasts"]:
            h = f["horizon_min"]
            _f(state, PREDICTION, "clinical", f"FORECAST_{h}MIN",
               f"Model forecast for {f['target_time']} (+{h} min): {f['predicted_mgdl']} mg/dL "
               f"(90% interval {f['lower_90_mgdl']}-{f['upper_90_mgdl']}).")
        for kind, below, thr, bound_key, rule_prefix in (("HYPO", True, HYPO, "lower_90_mgdl", "hypo"),
                                                          ("HYPER", False, SEVERE_HYPER, "upper_90_mgdl", "hyper")):
            levels = {f["horizon_min"]: risk_level(f["predicted_mgdl"], f[bound_key], below, thr) for f in d["forecasts"]}
            worst_h, worst = max(levels.items(), key=lambda kv: ("not elevated", "elevated", "high").index(kv[1]))
            name = "hypoglycaemia (<70 mg/dL)" if below else "level 2 hyperglycaemia (>250 mg/dL)"
            perf = ""
            if risk_perf and worst != "not elevated":
                rule = f"{rule_prefix}_if_{'point' if worst == 'high' else ('lower' if below else 'upper') + '_bound'}_{'below_70' if below else 'above_250'}"
                p = risk_perf.get(rule, {}).get(f"{worst_h}min")
                if p:
                    perf = (f" In validation this rule caught {p['sensitivity_pct']}% of such events "
                            f"with {p['precision_pct']}% precision.")
            why = {"high": f"the point forecast is beyond the threshold at +{worst_h} min",
                   "elevated": f"the 90% interval reaches the threshold at +{worst_h} min",
                   "not elevated": "neither the forecast nor its 90% interval reaches the threshold"}[worst]
            sev = {"high": "critical", "elevated": "warning", "not elevated": "info"}[worst]
            gc[f"{kind.lower()}_risk"] = worst
            _f(state, PREDICTION, "clinical", f"{kind}_RISK",
               f"Risk of {name} within 60 min: {worst.upper()} ({why}).{perf}", sev, levels=levels)
        effects = list(d["forecasts"][-1]["factor_effects_mgdl"].items())[:3]
        drivers = ", ".join(f"{g} {v:+} mg/dL" for g, v in effects)
        _f(state, PREDICTION, "context", "FORECAST_DRIVERS",
           f"Largest factor effects on the 60-min forecast, relative to an average moment in the cohort: {drivers}.")
        if d.get("recording_quality_flag"):
            _f(state, PREDICTION, "data_quality", "SENSOR_BIAS_SUSPECTED",
               "This recording is flagged for suspected CGM sensor bias (>10% of readings <54 mg/dL with low-risk therapy); "
               "interpret forecasts with caution.", "warning")

    if acc := state.last_ok("dm_get_forecast_accuracy"):
        d = acc.data
        o, base = d["overall"], d["baseline_persistence"]
        parts = [f"+{h}: RMSE {o[f'{h}min']['rmse']} mg/dL, MAE {o[f'{h}min']['mae']}, Clarke zone A {o[f'{h}min']['clarke_a_pct']}% "
                 f"(no-change baseline RMSE {base[f'{h}min']['rmse']})" for h in (30, 60)]
        _f(state, PREDICTION, "context", "FORECAST_ACCURACY",
           f"{d['validation']} on {d['trained_on']['patients']} patients: " + "; ".join(parts) + ".")
        if d["this_patient"]:
            tp = d["this_patient"]
            _f(state, PREDICTION, "context", "FORECAST_ACCURACY_PATIENT",
               f"For this patient (out-of-sample): +30 min MAE {tp['30min']['mae']} mg/dL (n={tp['30min']['n']}), "
               f"+60 min MAE {tp['60min']['mae']} mg/dL.")

    if prof := state.last_ok("dm_get_patient_profile"):
        d = prof.data
        a1c = d["labs"]["hba1c_pct"]
        if a1c is not None:
            above = a1c >= HBA1C_TARGET_PCT
            _f(state, CONTEXT, "clinical", "HBA1C",
               f"HbA1c {a1c}% ({d['labs']['hba1c_mmol_mol']:.0f} mmol/mol), "
               f"{'at or above' if above else 'below'} the common {HBA1C_TARGET_PCT}% target "
               f"(recording {d['labs_from_recording']}).", "warning" if above else "info")
        else:
            _f(state, CONTEXT, "data_quality", "HBA1C_MISSING", "HbA1c was not recorded for this patient.", "warning")
        if d["labs_not_recorded"]:
            _f(state, CONTEXT, "data_quality", "LABS_MISSING", "Not recorded: " + ", ".join(d["labs_not_recorded"]) + ".")
        _f(state, CONTEXT, "context", "HYPO_HISTORY_FIELD",
           f"Summary field 'hypoglycaemia history' = {d['hypoglycemia_history_field']}; this is a clinical-history field, "
           "not a CGM finding.")

    if meds := state.last_ok("dm_get_medication_events"):
        s = meds.data["summary_by_route"]
        _f(state, CONTEXT, "context", "MED_SUMMARY",
           f"In the {meds.data['window_hours']} h before {meds.data['as_of']}: " +
           "; ".join(f"{k} x{v['events']}" + (f" ({v['total_dose']:g} {v['unit']} total)" if v["total_dose"] else "")
                     for k, v in s.items()) + ".")

    if meals := state.last_ok("dm_get_meal_responses"):
        s = meals.data["summary"]
        if s["median_rise_mgdl"] is not None:
            _f(state, CONTEXT, "clinical", "MEAL_RESPONSE",
               f"Across the last {s['meals']} meals: median post-meal rise {s['median_rise_mgdl']} mg/dL; "
               f"{s['meals_peaking_above_180']} peaked above 180 mg/dL; {s['meals_followed_by_lt70']} were followed by a reading <70 mg/dL within 2 h.")

    if pat := state.last_ok("dm_find_similar_patterns"):
        s = pat.data["summary"]
        _f(state, CONTEXT, "clinical", "SIMILAR_PATTERNS",
           f"{s['matches']} most similar earlier 2-h windows (of {pat.data['history_searched']['windows_compared']} compared): "
           f"{s['followed_by_lt70']} were followed by glucose <70 mg/dL within 60 min; median 60-min change "
           f"{s['median_change_60min_mgdl']:+} mg/dL.", "warning" if s["followed_by_lt70"] else "info")

    if mem := state.last_ok("recall_session_memory"):
        d = mem.data
        cur = d["this_session"]
        _f(state, CONTEXT, "context", "SESSION_MEMORY",
           f"This conversation has retrieved {len(cur['facts'])} fact set(s) about {d['patient_id']} across "
           f"{len(cur['earlier_questions'])} earlier question(s); {len(d['earlier_sessions'])} question(s) about this "
           "patient were asked in this user's earlier sessions.")

    if gc:
        state.clinical_context["glucose"] = gc
