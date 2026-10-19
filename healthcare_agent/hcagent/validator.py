"""Result validator: checks tool outputs, fills structured context, and runs the rules.

Everything the LLM is later allowed to assert about thresholds, crowding or clinical
status comes from findings produced here.
"""

from __future__ import annotations

from typing import Any

from . import rules
from .data import ClinicalRepository, minutes_between
from .dm_rules import validate_dm
from .schemas import Finding, Intent, SharedState


def hm(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60:02d}m"


class ResultValidator:
    def __init__(self, repo: ClinicalRepository, dm_risk_rules: dict | None = None):
        self.repo = repo
        self.dm_risk_rules = dm_risk_rules  # validated performance of the glucose risk rules

    def validate(self, state: SharedState) -> None:
        """Recompute derived context from all tool results (idempotent; re-run after LLM tool calls)."""
        state.findings, state.patient = [], None
        state.clinical_context, state.hospital_context = {}, {}
        self._data_quality(state)
        self._patient_context(state)
        self._hospital_context(state)
        self._wait_context(state)
        validate_dm(state, self.dm_risk_rules)

    # ------------------------------------------------------------------ data quality

    def _data_quality(self, state: SharedState) -> None:
        for r in state.tool_results:
            if not r.ok:
                state.findings.append(Finding(
                    category="data_quality", code=r.error_code or "ERROR", message=f"{r.tool}: {r.error}",
                    severity="warning", evidence={"tool": r.tool, "args": r.args},
                ))
                continue
            if r.tool == "get_ed_status":
                cur = r.data["current"]
                lag = minutes_between(cur["timestamp"], r.data["requested_time"])
                if lag > rules.STALE_SNAPSHOT_MINUTES:
                    state.findings.append(Finding(
                        category="data_quality", code="STALE_ED_SNAPSHOT", severity="warning",
                        message=f"ED snapshot is {hm(lag)} older than the requested time.",
                    ))
                if not 0 <= cur["ed_occupancy_ratio"] <= 2:
                    state.findings.append(Finding(
                        category="data_quality", code="IMPLAUSIBLE_VALUE", severity="warning",
                        message=f"ED occupancy ratio {cur['ed_occupancy_ratio']} is outside 0-2.",
                    ))
            elif r.tool == "get_bed_capacity":
                for w in r.data["wards"]:
                    if w["occupied_beds"] + w["available_beds"] + w["blocked_beds"] != w["total_beds"]:
                        state.findings.append(Finding(
                            category="data_quality", code="BED_COUNT_MISMATCH", severity="warning",
                            message=f"{w['ward']}: occupied+available+blocked does not equal total beds.",
                            evidence=w,
                        ))
            elif r.tool == "get_recent_vitals":
                for obs in r.data["observations"]:
                    bad = [k for k, (lo, hi) in rules.PLAUSIBLE_RANGES.items()
                           if obs.get(k) is not None and not lo <= obs[k] <= hi]
                    obs["implausible"] = bad
                    if bad:
                        state.findings.append(Finding(
                            category="data_quality", code="IMPLAUSIBLE_VITAL", severity="warning",
                            message=f"Observation {obs['vital_id']} has implausible {', '.join(bad)}; excluded from scoring.",
                        ))

    # ------------------------------------------------------------------ patient

    def _patient_context(self, state: SharedState) -> None:
        profile = state.last_ok("get_patient_profile")
        conditions: list[str] = []
        if profile:
            state.patient = profile.data
            conditions = profile.data["chronic_conditions"]
            state.clinical_context["chronic_conditions"] = conditions

        enc = state.last_ok("get_patient_encounters")
        if enc:
            encs = enc.data["encounters"]
            state.clinical_context["encounter_count"] = enc.data["count"]
            state.clinical_context["encounter_window_days"] = enc.data["window_days"]
            state.clinical_context["latest_encounter"] = encs[0]
            if encs[0]["status_at_now"] == "IN_ED":
                state.clinical_context["current_encounter"] = encs[0]

        vit = state.last_ok("get_recent_vitals")
        if not vit:
            return
        obs_list = [o for o in vit.data["observations"] if not o.get("implausible")]
        if not obs_list:
            return
        latest = obs_list[0]
        hours_old = minutes_between(latest["timestamp"], state.clock) / 60
        state.clinical_context["latest_vitals"] = {k: latest[k] for k in ("timestamp", *rules.VITAL_LABELS, "diastolic_bp_mmHg")}
        state.clinical_context["hours_since_latest_vitals"] = round(hours_old, 1)

        if hours_old > rules.STALE_VITALS_HOURS:
            state.findings.append(Finding(
                category="context", code="VITALS_NOT_CURRENT", severity="warning",
                message=(f"Most recent observations are from {latest['timestamp']} "
                         f"({round(hours_old / 24, 1)} days before now); they do not describe the current condition."),
            ))

        score = rules.news2_partial(latest, conditions)
        state.clinical_context["news2_partial"] = score
        sev = {"high": "critical", "medium": "warning", "low-medium": "warning", "low": "info"}[score["band"]]
        state.findings.append(Finding(
            category="clinical", code="NEWS2_PARTIAL", severity=sev,
            message=(f"Partial NEWS2 = {score['total']} ({score['band']} risk band) from observations at "
                     f"{latest['timestamp']}; SpO2 {score['spo2_scale']}; excludes consciousness and oxygen therapy."),
            evidence=score,
        ))
        for param, comp in score["components"].items():
            if comp >= 2:
                label, unit = rules.VITAL_LABELS[param]
                state.findings.append(Finding(
                    category="clinical", code="VITAL_ABNORMAL", severity="critical" if comp == 3 else "warning",
                    message=f"{label} {latest[param]}{unit} at {latest['timestamp']} (NEWS2 component score {comp}).",
                    evidence={"parameter": param, "value": latest[param], "component_score": comp},
                ))
        if "COPD" in conditions and 88 <= latest["spo2_pct"] <= 92:
            state.findings.append(Finding(
                category="clinical", code="SPO2_IN_COPD_TARGET", severity="info",
                message=f"SpO2 {latest['spo2_pct']}% is within the usual COPD target range of 88-92%.",
            ))

        if len(obs_list) >= 2:
            first = obs_list[-1]
            changes = {
                "spo2_pct": latest["spo2_pct"] - first["spo2_pct"],
                "heart_rate_bpm": latest["heart_rate_bpm"] - first["heart_rate_bpm"],
                "respiratory_rate_bpm": latest["respiratory_rate_bpm"] - first["respiratory_rate_bpm"],
            }
            worsening = []
            if changes["spo2_pct"] <= -3:
                worsening.append(f"SpO2 fell from {first['spo2_pct']}% to {latest['spo2_pct']}%")
            if changes["heart_rate_bpm"] >= 20:
                worsening.append(f"heart rate rose from {first['heart_rate_bpm']} to {latest['heart_rate_bpm']} bpm")
            if changes["respiratory_rate_bpm"] >= 6:
                worsening.append(f"respiratory rate rose from {first['respiratory_rate_bpm']} to {latest['respiratory_rate_bpm']}/min")
            state.findings.append(Finding(
                category="clinical", code="VITALS_TREND",
                severity="warning" if worsening else "info",
                message=(f"Between {first['timestamp']} and {latest['timestamp']} ({len(obs_list)} observation sets): "
                         + ("; ".join(worsening) if worsening else "no worsening trend in SpO2, heart rate or respiratory rate.")),
                evidence={"changes": changes, "n": len(obs_list)},
            ))

    # ------------------------------------------------------------------ hospital

    def _hospital_context(self, state: SharedState) -> None:
        ed = state.last_ok("get_ed_status")
        if ed:
            cur = ed.data["current"]
            hc = state.hospital_context
            hc.update({
                "ed_snapshot_time": cur["timestamp"],
                "ed_occupancy": cur["ed_occupancy_ratio"],
                "patients_in_ed": cur["patients_in_ed"],
                "patients_waiting": cur["patients_waiting"],
                "patients_on_trolleys": cur["patients_on_trolleys"],
                "average_wait_minutes": cur["average_wait_minutes"],
                "longest_wait_minutes": cur["longest_wait_minutes"],
                "ambulance_arrivals_last_hour": cur["ambulance_arrivals"],
                "ed_arrivals_last_hour": cur["ed_arrivals_last_hour"],
            })
            level, sev = rules.ed_crowding_level(cur["ed_occupancy_ratio"])
            hc["ed_crowding_level"] = level
            state.findings.append(Finding(
                category="operational", code="ED_CROWDING", severity=sev,
                message=(f"ED occupancy {cur['ed_occupancy_ratio']:.0%} ({level}) at {cur['timestamp']}: "
                         f"{cur['patients_in_ed']} in department, {cur['patients_waiting']} waiting, "
                         f"{cur['patients_on_trolleys']} on trolleys; average wait {cur['average_wait_minutes']} min."),
                evidence={"thresholds": {"busy": rules.ED_BUSY, "severe": rules.ED_SEVERE, "over": rules.ED_OVER_CAPACITY}},
            ))
            pct = self.repo.ed_percentile("ed_occupancy_ratio", cur["ed_occupancy_ratio"], cur["timestamp"])
            if pct is not None:
                hc["ed_occupancy_percentile_30d"] = pct
                state.findings.append(Finding(
                    category="operational", code="ED_RELATIVE_PRESSURE",
                    message=f"This occupancy is at the {pct}th percentile of hourly snapshots over the previous 30 days.",
                ))
            if cur["longest_wait_minutes"] > rules.FOUR_HOUR_MINUTES:
                state.findings.append(Finding(
                    category="operational", code="FOUR_HOUR_EXCEEDED", severity="warning",
                    message=f"Longest current ED wait is {cur['longest_wait_minutes']} min ({hm(cur['longest_wait_minutes'])}), above 4 hours.",
                ))
            trend = ed.data["trend"]
            if trend:
                old = trend[-1]
                state.findings.append(Finding(
                    category="operational", code="ED_TREND",
                    message=(f"From {old['timestamp']} to {cur['timestamp']}: occupancy {old['ed_occupancy_ratio']:.0%} -> "
                             f"{cur['ed_occupancy_ratio']:.0%}, waiting {old['patients_waiting']} -> {cur['patients_waiting']}, "
                             f"average wait {old['average_wait_minutes']} -> {cur['average_wait_minutes']} min."),
                    evidence={"hours": len(trend)},
                ))

        beds = state.last_ok("get_bed_capacity")
        if beds:
            t = beds.data["totals"]
            wards = beds.data["wards"]
            state.hospital_context.update({
                "beds_snapshot_time": wards[0]["timestamp"],
                "available_beds_total": t["available_beds"],
                "bed_occupancy_total": t["occupancy_ratio"],
                "available_beds_by_ward": {w["ward"]: w["available_beds"] for w in wards},
            })
            state.findings.append(Finding(
                category="operational", code="BED_TOTALS",
                message=(f"{t['available_beds']} of {t['total_beds']} inpatient beds available across "
                         f"{len(wards)} ward(s) at {wards[0]['timestamp']} ({t['occupancy_ratio']:.0%} occupied, "
                         f"{t['blocked_beds']} blocked)."),
            ))
            for w in wards:
                pressure = rules.ward_pressure(w)
                if pressure:
                    state.findings.append(Finding(
                        category="operational", code="WARD_PRESSURE",
                        severity="critical" if pressure == "full" else "warning",
                        message=(f"{w['ward']} ward {pressure}: {w['available_beds']} of {w['total_beds']} beds available "
                                 f"({w['occupancy_ratio']:.0%} occupied, {w['blocked_beds']} blocked)."),
                        evidence={"ward": w["ward"]},
                    ))

    # ------------------------------------------------------------------ explanation rules

    def _wait_context(self, state: SharedState) -> None:
        if state.parsed is None or Intent.PATIENT_ED_STATUS not in state.parsed.intents:
            return
        enc: dict[str, Any] | None = state.clinical_context.get("latest_encounter")
        if enc is None:
            return
        in_ed = enc["status_at_now"] == "IN_ED"
        if in_ed:
            msg = (f"Patient arrived at {enc['arrival_time']} with {enc['chief_complaint']} (triage category "
                   f"{enc['triage_level']}) and has been in the ED for {hm(enc['ed_minutes'])} as of {state.clock}.")
        else:
            msg = (f"Patient is not currently in the ED. Latest encounter {enc['encounter_id']}: arrived "
                   f"{enc['arrival_time']} ({enc['chief_complaint']}, triage {enc['triage_level']}), left the ED at "
                   f"{enc['ed_departure_time']} ({enc['disposition']}) after {hm(enc['ed_minutes'])}.")
        state.findings.append(Finding(category="context", code="ENCOUNTER_STATUS", message=msg,
                                      evidence={"in_ed": in_ed, "ed_minutes": enc["ed_minutes"]}))

        hc = state.hospital_context
        ops_factors: list[str] = []
        if "ed_occupancy" in hc:
            if hc["ed_occupancy"] >= rules.ED_SEVERE:
                ops_factors.append(f"ED occupancy {hc['ed_occupancy']:.0%}")
            if hc.get("ed_occupancy_percentile_30d", 0) >= 75:
                ops_factors.append(f"busier than usual ({hc['ed_occupancy_percentile_30d']}th percentile for the past 30 days)")
            if enc["ed_minutes"] > hc["average_wait_minutes"]:
                ops_factors.append(
                    f"this stay ({enc['ed_minutes']} min) exceeds the ED average wait of "
                    f"{hc['average_wait_minutes']} min at {hc['ed_snapshot_time']}")
        pressured = [w for w, n in hc.get("available_beds_by_ward", {}).items() if n <= 2]
        if pressured:
            ops_factors.append("2 or fewer beds available on: " + ", ".join(pressured))
        if enc["triage_level"] >= 3:
            ops_factors.append(f"triage category {enc['triage_level']} is prioritised after categories 1-2")

        score = state.clinical_context.get("news2_partial")
        if score is None:
            clinical = "No observations were available to assess clinical status."
            stable = None
        else:
            stable = score["band"] == "low"
            clinical = (f"Partial NEWS2 {score['total']} ({score['band']}) at {score['observation_time']}: "
                        + ("no deterioration signal." if stable else "abnormal observations present."))

        if stable is False:
            code, sev = "CLINICAL_PRIORITY", "critical" if score["band"] in ("high", "medium") else "warning"
            summary = "Abnormal observations: clinical review takes priority over explaining the wait. " + clinical
        elif ops_factors:
            code, sev = "WAIT_DRIVERS_OPERATIONAL", "info"
            summary = ("The available evidence is consistent with operational pressure rather than clinical "
                       "deterioration as the main driver. " + clinical)
        else:
            code, sev = "WAIT_DRIVERS_UNCLEAR", "info"
            summary = "Neither operational pressure nor clinical deterioration in the available data explains the wait. " + clinical
        state.findings.append(Finding(
            category="context", code=code, severity=sev, message=summary,
            evidence={"operational_factors": ops_factors, "clinically_stable": stable},
        ))
