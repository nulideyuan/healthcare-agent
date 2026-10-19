"""Deterministic responses for routes that must not depend on the LLM, and an offline fallback."""

from __future__ import annotations

from .schemas import Role, SharedState


def denied(state: SharedState) -> str:
    reason = state.permissions[-1].reason if state.permissions else "Not permitted."
    return f"Access denied. {reason} No patient data was retrieved for this request."


def clarify(state: SharedState) -> str:
    return ("Which patient do you mean? Please include the patient ID (for example P0042, or DM2045 for the diabetes cohort). "
            "No data was retrieved for this request.")


def not_found(state: SharedState) -> str:
    return f"No patient with ID {state.target_patient_id} was found in the record. No data was retrieved."


def unsupported(state: SharedState) -> str:
    items = " and ".join(state.unsupported) or "the requested information"
    msg = (f"{items[0].upper() + items[1:]} are not available in this system (v1 dataset), "
           "so I cannot report them and have not estimated any values.")
    if state.user.role == Role.PATIENT:
        msg += " Please ask your care team for this information."
    return msg


def evidence_summary(state: SharedState) -> str:
    """Plain rendering of the validated evidence, used when no LLM is configured or it fails."""
    lines: list[str] = []
    if state.patient:
        p = state.patient
        conds = ", ".join(p["chronic_conditions"]) or "none recorded"
        lines.append(f"Patient {p['patient_id']}: {p['age']}-year-old {p['sex'].lower()}, "
                     f"blood group {p['blood_group']}, BMI {p['bmi']}; chronic conditions: {conds}.")
    enc = next((r for r in reversed(state.tool_results) if r.tool == "get_patient_encounters" and r.ok), None)
    if enc and not any(f.code == "ENCOUNTER_STATUS" for f in state.findings):
        lines.append(f"{enc.data['count']} encounter(s) found"
                     + (f" in the last {enc.data['window_days']} days" if enc.data["window_days"] else "") + ":")
        for e in enc.data["encounters"]:
            lines.append(f"  - {e['arrival_time']} {e['chief_complaint']} (triage {e['triage_level']}), "
                         f"{e['disposition'] or 'still in ED'}")
    v = state.clinical_context.get("latest_vitals")
    if v:
        lines.append(f"Latest observations ({v['timestamp']}): HR {v['heart_rate_bpm']} bpm, RR "
                     f"{v['respiratory_rate_bpm']}/min, SpO2 {v['spo2_pct']}%, temperature {v['temperature_c']}°C, "
                     f"BP {v['systolic_bp_mmHg']}/{v['diastolic_bp_mmHg']} mmHg.")
    order = {"context": 0, "clinical": 1, "operational": 2, "data_quality": 3}
    for f in sorted(state.findings, key=lambda f: order[f.category]):
        lines.append(f"[{f.category}] {f.message}")
    if state.unsupported:
        lines.append(unsupported(state))
    return "\n".join(lines) if lines else "No information was found for this request."
