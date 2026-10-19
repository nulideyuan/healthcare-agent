"""Train and evaluate the 30/60-min glucose forecaster on ShanghaiT2DM.

    python3 diabetes/train_forecaster.py [--deploy auto|ridge|lgbm]

Evaluation is patient-grouped 5-fold cross-validation: a patient's recordings are never in both the
training and the test fold. Every patient gets out-of-fold (OOF) predictions, and at runtime each
patient is served by the fold model that never saw them, so the accuracy the agent reports is
honest. Recordings flagged as suspected sensor bias are excluded from training and evaluation.

Compared: persistence (no change), linear trend extrapolation, ridge regression, LightGBM.
Writes data/models/glucose_forecaster_v1.json.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from hcagent.dm_features import FEATURES, HORIZONS_MIN, RecordingData, build_features, clarke_zone, targets  # noqa: E402

DB = PROJECT / "data" / "shanghai_t2dm.sqlite"
OUT = PROJECT / "data" / "models" / "glucose_forecaster_v1.json"
N_FOLDS = 5
SEED = 42
RIDGE_ALPHA = 10.0
G_BINS = [0, 100, 180, 1000]   # interval calibration bins on current glucose


def load_recordings(con) -> dict[str, tuple[str, str | None, RecordingData]]:
    con.row_factory = sqlite3.Row
    recs = {}
    for r in con.execute("SELECT * FROM dm_recordings ORDER BY recording_id"):
        rid = r["recording_id"]
        cgm = con.execute("SELECT ts, glucose_mgdl FROM dm_cgm WHERE recording_id=? ORDER BY ts", (rid,)).fetchall()
        meals = [(m["ts"], m["total_grams"]) for m in con.execute(
            "SELECT ts, total_grams FROM dm_meals WHERE recording_id=? ORDER BY ts", (rid,))]
        meds = [(m["ts"], m["route"], m["insulin_type"], m["dose"]) for m in con.execute(
            "SELECT ts, route, insulin_type, dose FROM dm_medications WHERE recording_id=? ORDER BY ts", (rid,))]
        static = {k: r[k] for k in ("age", "bmi", "hba1c_pct", "diabetes_duration_years", "sex")}
        recs[rid] = (r["patient_id"], r["quality_flag"],
                     RecordingData([c[0] for c in cgm], [c[1] for c in cgm], meals, meds, static))
    return recs


# --------------------------------------------------------------------------- models


def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {"mean": mu.tolist(), "std": sd.tolist(), "coef": w.tolist(), "intercept": float(y.mean())}


def ridge_predict(m, X):
    return ((X - np.array(m["mean"])) / np.array(m["std"])) @ np.array(m["coef"]) + m["intercept"]


def lgbm_fit(X, y):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    return model.fit(X, y)


# --------------------------------------------------------------------------- metrics


def metrics(ref, pred):
    err = pred - ref
    zones = np.array([clarke_zone(r, p) for r, p in zip(ref, pred)])
    hypo_true, hypo_pred = ref < 70, pred < 70
    tp = int((hypo_true & hypo_pred).sum())
    return {
        "n": int(len(ref)),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "mard_pct": round(float(np.mean(np.abs(err) / ref) * 100), 2),
        "clarke_a_pct": round(float((zones == "A").mean() * 100), 2),
        "clarke_ab_pct": round(float(np.isin(zones, ["A", "B"]).mean() * 100), 2),
        "hypo_sensitivity_pct": round(100 * tp / max(int(hypo_true.sum()), 1), 1),
        "hypo_precision_pct": round(100 * tp / max(int(hypo_pred.sum()), 1), 1),
        "hypo_events_in_test": int(hypo_true.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", choices=["auto", "ridge", "lgbm"], default="auto")
    args = ap.parse_args()
    t0 = time.time()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    recs = load_recordings(con)
    excluded = sorted(rid for rid, (_, flag, _) in recs.items() if flag)

    Xs, Ys, G0, P, R = [], [], [], [], []
    for rid, (pid, flag, rec) in recs.items():
        if flag:
            continue
        X, valid, t = build_features(rec)
        Y, yv = targets(rec, t)
        m = valid & yv
        Xs.append(X[m]); Ys.append(Y[m]); G0.append(X[m, 0]); P += [pid] * int(m.sum()); R += [rid] * int(m.sum())
    X, Y, g0 = np.vstack(Xs), np.vstack(Ys), np.concatenate(G0)
    P, R = np.array(P), np.array(R)

    patients = np.array(sorted(set(P)))
    rng = np.random.default_rng(SEED)
    fold_of = {p: i % N_FOLDS for i, p in enumerate(rng.permutation(patients))}
    fold = np.array([fold_of[p] for p in P])
    print(f"samples {len(X)}, patients {len(patients)}, excluded recordings {excluded}")

    systems = ["persistence", "linear_trend", "ridge", "lgbm"]
    oof = {s: np.full_like(Y, np.nan) for s in systems}
    fold_models = {"ridge": [], "lgbm": []}
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        ridge_h, lgbm_h = [], []
        for j, h in enumerate(HORIZONS_MIN):
            oof["persistence"][te, j] = 0.0
            oof["linear_trend"][te, j] = X[te, FEATURES.index("slope30")] * h
            rm = ridge_fit(X[tr], Y[tr, j]); ridge_h.append(rm)
            oof["ridge"][te, j] = ridge_predict(rm, X[te])
            lm = lgbm_fit(X[tr], Y[tr, j]); lgbm_h.append(lm)
            oof["lgbm"][te, j] = lm.predict(X[te])
        fold_models["ridge"].append(ridge_h); fold_models["lgbm"].append(lgbm_h)
        print(f"fold {f} done ({te.sum()} test samples)")

    results = {}
    for s in systems:
        results[s] = {f"{h}min": metrics(g0 + Y[:, j], g0 + oof[s][:, j]) for j, h in enumerate(HORIZONS_MIN)}
    print(f"\n{'model':<14}" + "".join(f"{h}min RMSE / MAE / ClarkeA / hypoSens   " for h in HORIZONS_MIN))
    for s in systems:
        print(f"{s:<14}" + "".join(
            f"{results[s][f'{h}min']['rmse']:>6} / {results[s][f'{h}min']['mae']:>5} / {results[s][f'{h}min']['clarke_a_pct']:>5}% / "
            f"{results[s][f'{h}min']['hypo_sensitivity_pct']:>5}%      " for h in HORIZONS_MIN))

    if args.deploy == "auto":
        gain = 1 - results["lgbm"]["60min"]["rmse"] / results["ridge"]["60min"]["rmse"]
        deploy = "lgbm" if gain >= 0.03 else "ridge"
        print(f"\nLightGBM vs ridge RMSE gain at 60 min: {gain:.1%} -> deploying {deploy}")
    else:
        deploy = args.deploy

    # Prediction intervals: empirical 5th/95th percentile of OOF residuals, by current-glucose bin.
    intervals = {}
    for j, h in enumerate(HORIZONS_MIN):
        resid = Y[:, j] - oof[deploy][:, j]
        intervals[f"{h}min"] = [{"g_low": lo, "g_high": hi,
                                 "q05": round(float(np.quantile(resid[(g0 >= lo) & (g0 < hi)], 0.05)), 2),
                                 "q95": round(float(np.quantile(resid[(g0 >= lo) & (g0 < hi)], 0.95)), 2)}
                                for lo, hi in zip(G_BINS[:-1], G_BINS[1:])]
        cover = np.mean([(lo_q <= r <= hi_q) for r, gg in zip(resid, g0)
                         for b in intervals[f"{h}min"] if b["g_low"] <= gg < b["g_high"]
                         for lo_q, hi_q in [(b["q05"], b["q95"])]])
        print(f"{h}min 90% interval empirical coverage: {cover:.1%}")

    # Risk rules use the interval bounds, because point forecasts regress towards the mean and miss lows.
    risk_rules = {}
    for j, h in enumerate(HORIZONS_MIN):
        q = np.zeros((len(g0), 2))
        for b in intervals[f"{h}min"]:
            m = (g0 >= b["g_low"]) & (g0 < b["g_high"])
            q[m] = (b["q05"], b["q95"])
        point = g0 + oof[deploy][:, j]
        actual = g0 + Y[:, j]
        for name, flag, truth in (("hypo_if_lower_bound_below_70", point + q[:, 0] < 70, actual < 70),
                                  ("hypo_if_point_below_70", point < 70, actual < 70),
                                  ("hyper_if_upper_bound_above_250", point + q[:, 1] > 250, actual > 250),
                                  ("hyper_if_point_above_250", point > 250, actual > 250)):
            tp = int((flag & truth).sum())
            risk_rules.setdefault(name, {})[f"{h}min"] = {
                "sensitivity_pct": round(100 * tp / max(int(truth.sum()), 1), 1),
                "precision_pct": round(100 * tp / max(int(flag.sum()), 1), 1),
                "flag_rate_pct": round(100 * float(flag.mean()), 2), "events": int(truth.sum())}
    print("risk rules (OOF):", json.dumps(risk_rules, indent=1))

    per_patient = {}
    for p in patients:
        m = P == p
        per_patient[p] = {f"{h}min": {k: v for k, v in metrics(g0[m] + Y[m, j], g0[m] + oof[deploy][m, j]).items()
                                      if k in ("n", "rmse", "mae", "mard_pct", "clarke_a_pct")}
                          for j, h in enumerate(HORIZONS_MIN)}

    if deploy == "ridge":
        serialised = [[m for m in fm] for fm in fold_models["ridge"]]
    else:
        serialised = [[m.booster_.model_to_string() for m in fm] for fm in fold_models["lgbm"]]

    artifact = {
        "model": "glucose_forecaster", "version": "1.0", "algorithm": deploy,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {"dataset": "ShanghaiT2DM", "samples": int(len(X)), "patients": int(len(patients)),
                 "excluded_recordings": excluded, "cgm_step_min": 15},
        "features": FEATURES, "horizons_min": list(HORIZONS_MIN), "target": "glucose change (mg/dL) from current reading",
        "validation": f"patient-grouped {N_FOLDS}-fold cross-validation (out-of-fold)",
        "patient_fold": {p: int(fold_of[p]) for p in patients},
        "fold_models": serialised,
        "intervals_90pct": intervals,
        "risk_rules_oof": risk_rules,
        "metrics_oof": results,
        "metrics_per_patient": per_patient,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
