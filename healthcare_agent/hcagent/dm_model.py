"""Runtime glucose forecaster: loads the trained artifact and predicts 30/60 min ahead.

Each patient is served by the cross-validation fold model that never saw that patient, so the
forecast is out-of-sample and matches the reported validation accuracy. This is pure numpy, so the
agent venv doesn't need scikit-learn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .dm_features import FEATURES, LAGS, RecordingData, build_features

# Correlated lag features cancel each other in a linear model, so effects are reported per factor group.
FACTOR_GROUPS = {
    "recent CGM trajectory (last 2 h)": ["g0", *[f"d{k}" for k in range(1, LAGS + 1)], "slope15", "slope30", "slope60",
                                         "accel", "mean2h", "sd2h", "min2h", "max2h"],
    "time of day": ["tod_sin", "tod_cos"],
    "recent meals": ["meal_g_2h", "meal_g_4h", "min_since_meal", "meal_last_30"],
    "insulin": ["rapid_iu_4h", "basal_sc_iu_24h", "pump_basal_rate"],
    "oral agents": ["oral_6h"],
    "patient characteristics (age, BMI, HbA1c, duration, sex)": ["age", "bmi", "hba1c_pct", "diabetes_duration_years", "male"],
}
_GROUP_IDX = {g: [FEATURES.index(f) for f in fs] for g, fs in FACTOR_GROUPS.items()}


class GlucoseForecaster:
    def __init__(self, path: Path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}. Run: python3 diabetes/train_forecaster.py")
        self.a = json.loads(Path(path).read_text())
        if self.a["algorithm"] != "ridge":
            raise ValueError("The runtime supports the ridge artifact only (pure numpy).")
        self.name = f"{self.a['model']}_{self.a['algorithm']}"
        self.version = self.a["version"]
        self.horizons = self.a["horizons_min"]

    def fold_for(self, patient_id: str) -> int:
        # Patients excluded from training (flagged recordings) are out-of-sample for every fold.
        return self.a["patient_fold"].get(patient_id, 0)

    def _interval(self, horizon: int, g0: float) -> tuple[float, float]:
        for b in self.a["intervals_90pct"][f"{horizon}min"]:
            if b["g_low"] <= g0 < b["g_high"]:
                return b["q05"], b["q95"]
        raise ValueError(g0)

    def predict(self, patient_id: str, rec: RecordingData) -> dict[str, Any] | None:
        """Forecast from the last reading in `rec`. None if the last 2 h are not a regular CGM series."""
        X, valid, _ = build_features(rec)
        if len(X) == 0 or not valid[-1]:
            return None
        x = X[-1]
        g0 = float(x[0])
        fold = self.fold_for(patient_id)
        out = []
        for j, h in enumerate(self.horizons):
            m = self.a["fold_models"][fold][j]
            z = (x - np.array(m["mean"])) / np.array(m["std"])
            contrib = z * np.array(m["coef"])
            delta = float(contrib.sum() + m["intercept"])
            lo, hi = self._interval(h, g0)
            out.append({
                "horizon_min": h,
                "predicted_mgdl": round(g0 + delta, 1),
                "lower_90_mgdl": round(g0 + delta + lo, 1),
                "upper_90_mgdl": round(g0 + delta + hi, 1),
                "predicted_change_mgdl": round(delta, 1),
                # Effect of each factor group relative to an average patient-moment (mg/dL, sums to the change
                # minus the average change).
                "factor_effects_mgdl": dict(sorted(((g, round(float(contrib[idx].sum()), 1)) for g, idx in _GROUP_IDX.items()),
                                                   key=lambda kv: -abs(kv[1]))),
            })
        return {"current_mgdl": g0, "fold": fold, "forecasts": out,
                "inputs": {k: round(float(x[FEATURES.index(k)]), 2) for k in
                           ("slope30", "meal_g_2h", "min_since_meal", "rapid_iu_4h", "pump_basal_rate", "oral_6h")}}

    def accuracy(self, patient_id: str) -> dict[str, Any]:
        a = self.a
        return {
            "model": self.name, "version": self.version, "validation": a["validation"],
            "trained_on": a["data"], "overall": a["metrics_oof"][a["algorithm"]],
            "baseline_persistence": a["metrics_oof"]["persistence"],
            "this_patient": a["metrics_per_patient"].get(patient_id),
            "risk_rules": a["risk_rules_oof"],
            "interval": "90% empirical interval from out-of-fold residuals (coverage 90% in validation)",
        }
