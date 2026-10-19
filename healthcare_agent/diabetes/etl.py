"""ETL: ShanghaiT2DM raw Excel files -> data/shanghai_t2dm.sqlite (read-only at runtime).

    python3 diabetes/etl.py [--raw ~/Downloads/diabetes_datasets]

Needs pandas, openpyxl and xlrd (for the legacy .xls files). Run with the analysis Python, not the
agent venv. Re-running rebuilds the database from scratch.

Tables
  dm_patients      one row per patient (DM2001 ...)
  dm_recordings    one row per monitoring period: demographics, labs, agents, quality flags
  dm_cgm           15-min CGM readings
  dm_cbg           finger-stick glucose
  dm_meals         dietary intake entries (EN + ZH text, parsed total grams)
  dm_medications   insulin (s.c., pump bolus/basal, i.v.) and oral agents, with parsed doses
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = Path.home() / "Downloads" / "diabetes_datasets"
DB_PATH = PROJECT / "data" / "shanghai_t2dm.sqlite"

# 5 header layouts exist in the raw files; the 11 columns always appear in this order.
CANONICAL = ["time", "cgm", "cbg", "ketone", "meal_en", "meal_zh", "insulin_sc",
             "oral_agents", "csii_bolus", "csii_basal", "insulin_iv"]

SUMMARY_COLUMNS = {
    "Gender (Female=1, Male=2)": "sex",
    "Age (years)": "age",
    "Height (m)": "height_m",
    "Weight (kg)": "weight_kg",
    "BMI (kg/m2)": "bmi",
    "Smoking History (pack year)": "smoking_pack_years",
    "Alcohol Drinking History (drinker/non-drinker)": "alcohol",
    "Duration of diabetes (years)": "diabetes_duration_years",
    "Acute Diabetic Complications": "acute_complications",
    "Diabetic Macrovascular Complications": "macrovascular_complications",
    "Diabetic Microvascular Complications": "microvascular_complications",
    "Comorbidities": "comorbidities",
    "Hypoglycemic Agents": "hypoglycemic_agents",
    "Other Agents": "other_agents",
    "Fasting Plasma Glucose (mg/dl)": "fasting_glucose_mgdl",
    "2-hour Postprandial Plasma Glucose (mg/dl)": "postprandial_2h_glucose_mgdl",
    "Fasting C-peptide (nmol/L)": "fasting_c_peptide_nmol_l",
    "2-hour Postprandial C-peptide (nmol/L)": "postprandial_2h_c_peptide_nmol_l",
    "Fasting Insulin (pmol/L)": "fasting_insulin_pmol_l",
    "2-hour Postprandial insulin (pmol/L)": "postprandial_2h_insulin_pmol_l",
    "HbA1c (mmol/mol)": "hba1c_mmol_mol",
    "Glycated Albumin (%)": "glycated_albumin_pct",
    "Total Cholesterol (mmol/L)": "total_cholesterol_mmol_l",
    "Triglyceride (mmol/L)": "triglyceride_mmol_l",
    "High-Density Lipoprotein Cholesterol (mmol/L)": "hdl_mmol_l",
    "Low-Density Lipoprotein Cholesterol (mmol/L)": "ldl_mmol_l",
    "Creatinine (umol/L)": "creatinine_umol_l",
    "Estimated Glomerular Filtration Rate (ml/min/1.73m2)": "egfr_ml_min",
    "Uric Acid (mmol/L)": "uric_acid",
    "Blood Urea Nitrogen (mmol/L)": "bun_mmol_l",
    "Hypoglycemia (yes/no)": "hypoglycemia_history",
}

RAPID = ("aspart,", "aspart ", "glulisine", "lispro", "novolin r", "humulin r", "gansulin r")
PREMIXED = ("70/30", "30r", "40r", "50r", "25r", "mix", "m30")
BASAL = ("degludec", "glargine", "glarigine", "detemir", "nph", "humulin n", "novolin n")
SENSOR_BIAS_TBR54_PCT = 10.0

SCHEMA = """
CREATE TABLE dm_patients (
    patient_id TEXT PRIMARY KEY, source_number TEXT NOT NULL, sex TEXT, n_recordings INTEGER
);
CREATE TABLE dm_recordings (
    recording_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, source_file TEXT, start_date TEXT,
    cgm_start TEXT, cgm_end TEXT, n_cgm INTEGER, completeness_pct REAL, max_gap_min REAL,
    quality_flag TEXT, sex TEXT, age INTEGER, height_m REAL, weight_kg REAL, bmi REAL,
    smoking_pack_years REAL, alcohol TEXT, diabetes_duration_years REAL, acute_complications TEXT,
    macrovascular_complications TEXT, microvascular_complications TEXT, comorbidities TEXT,
    hypoglycemic_agents TEXT, other_agents TEXT, fasting_glucose_mgdl REAL,
    postprandial_2h_glucose_mgdl REAL, fasting_c_peptide_nmol_l REAL, postprandial_2h_c_peptide_nmol_l REAL,
    fasting_insulin_pmol_l REAL, postprandial_2h_insulin_pmol_l REAL, hba1c_mmol_mol REAL, hba1c_pct REAL,
    glycated_albumin_pct REAL, total_cholesterol_mmol_l REAL, triglyceride_mmol_l REAL, hdl_mmol_l REAL,
    ldl_mmol_l REAL, creatinine_umol_l REAL, egfr_ml_min REAL, uric_acid REAL, bun_mmol_l REAL,
    hypoglycemia_history TEXT,
    FOREIGN KEY(patient_id) REFERENCES dm_patients(patient_id)
);
CREATE TABLE dm_cgm (
    recording_id TEXT NOT NULL, patient_id TEXT NOT NULL, ts TEXT NOT NULL, glucose_mgdl REAL NOT NULL,
    PRIMARY KEY (recording_id, ts)
);
CREATE TABLE dm_cbg (
    recording_id TEXT NOT NULL, patient_id TEXT NOT NULL, ts TEXT NOT NULL, glucose_mgdl REAL NOT NULL
);
CREATE TABLE dm_meals (
    meal_id TEXT PRIMARY KEY, recording_id TEXT NOT NULL, patient_id TEXT NOT NULL, ts TEXT NOT NULL,
    description_en TEXT, description_zh TEXT, total_grams REAL
);
CREATE TABLE dm_medications (
    med_id TEXT PRIMARY KEY, recording_id TEXT NOT NULL, patient_id TEXT NOT NULL, ts TEXT NOT NULL,
    route TEXT NOT NULL, insulin_type TEXT, description TEXT, dose REAL, unit TEXT
);
CREATE INDEX idx_cgm_patient_ts ON dm_cgm(patient_id, ts);
CREATE INDEX idx_cbg_patient_ts ON dm_cbg(patient_id, ts);
CREATE INDEX idx_meals_patient_ts ON dm_meals(patient_id, ts);
CREATE INDEX idx_meds_patient_ts ON dm_medications(patient_id, ts);
CREATE INDEX idx_rec_patient ON dm_recordings(patient_id);
"""


def ts(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M")


def clean_text(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = re.sub(r"\s+", " ", str(v).replace("\xa0", " ")).strip()
    return s or None


def insulin_type(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in PREMIXED):
        return "premixed"
    if any(k in d for k in BASAL):
        return "basal"
    if any(k in d for k in RAPID):
        return "rapid"
    return "unknown"


def parse_iu(desc: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*IU", desc, re.I)
    return float(m.group(1)) if m else None


def meal_grams(desc: str | None) -> float | None:
    if not desc:
        return None
    grams = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*g\b", desc)]
    return sum(grams) if grams else None


def load_summary(raw: Path) -> pd.DataFrame:
    s = pd.read_excel(raw / "Shanghai_T2DM_Summary.xlsx")
    s = s.mask(s.astype(str).apply(lambda c: c.str.strip()).eq("/"))
    s.columns = [re.sub(r"\s+", " ", c).strip() for c in s.columns]
    s = s.rename(columns=SUMMARY_COLUMNS)
    s["recording_id"] = s["Patient Number"].str.rsplit("_", n=1).str[0]
    numeric = [c for c in SUMMARY_COLUMNS.values() if c.endswith(("_mgdl", "_nmol_l", "_pmol_l", "_mmol_l", "_mmol_mol",
                                                                  "_pct", "_umol_l", "_ml_min", "_m", "_kg", "_years"))]
    numeric += ["bmi", "uric_acid", "age"]
    for c in numeric:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["hba1c_pct"] = (s["hba1c_mmol_mol"] / 10.929 + 2.15).round(1)
    s["sex"] = s["sex"].map({1: "female", 2: "male"})
    for c in s.columns:
        if s[c].dtype == object:
            s[c] = s[c].map(clean_text)
    return s


def build(raw: Path, db_path: Path) -> None:
    summary = load_summary(raw).set_index("Patient Number")
    files = sorted(p for p in (raw / "Shanghai_T2DM").iterdir() if p.suffix in (".xls", ".xlsx"))
    cgm_rows, cbg_rows, meal_rows, med_rows, rec_rows = [], [], [], [], []

    for path in files:
        stem = path.stem                                  # 2001_1_20201117
        rec_id = stem.rsplit("_", 1)[0]                   # 2001_1
        pid = "DM" + stem.split("_")[0]                   # DM2001
        df = pd.read_excel(path)
        df.columns = CANONICAL
        df["time"] = pd.to_datetime(df["time"])
        for c in ("cgm", "cbg"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        cgm = df.dropna(subset=["cgm"]).drop_duplicates("time").sort_values("time")
        cgm_rows += [(rec_id, pid, ts(t), round(float(g), 1)) for t, g in zip(cgm["time"], cgm["cgm"])]
        cbg_rows += [(rec_id, pid, ts(t), round(float(g), 1))
                     for t, g in df.dropna(subset=["cbg"])[["time", "cbg"]].itertuples(index=False)]

        for i, r in enumerate(df.itertuples(index=False)):
            when = ts(r.time)
            en, zh = clean_text(r.meal_en), clean_text(r.meal_zh)
            if en or zh:
                meal_rows.append((f"{rec_id}-M{i:04d}", rec_id, pid, when, en, zh, meal_grams(en)))
            sc = clean_text(r.insulin_sc)
            if sc:
                med_rows.append((f"{rec_id}-S{i:04d}", rec_id, pid, when, "insulin_sc", insulin_type(sc), sc, parse_iu(sc), "IU"))
            oral = clean_text(r.oral_agents)
            if oral:
                med_rows.append((f"{rec_id}-O{i:04d}", rec_id, pid, when, "oral", None, oral, None, None))
            for col, route, unit in (("csii_bolus", "insulin_pump_bolus", "IU"), ("csii_basal", "insulin_pump_basal", "IU/h")):
                v = pd.to_numeric(getattr(r, col), errors="coerce")
                if pd.notna(v):
                    med_rows.append((f"{rec_id}-{route[-5:].upper()}{i:04d}", rec_id, pid, when, route, "rapid",
                                     f"Novolin R {v:g} {unit}", float(v), unit))
            iv = clean_text(r.insulin_iv)
            if iv and parse_iu(iv) is not None:           # one cell holds a stray header string; skip it
                med_rows.append((f"{rec_id}-V{i:04d}", rec_id, pid, when, "insulin_iv", "rapid", iv, parse_iu(iv), "IU"))

        g = cgm["cgm"].to_numpy()
        dt = cgm["time"].diff().dt.total_seconds().div(60).dropna()
        span = (cgm["time"].iat[-1] - cgm["time"].iat[0]).total_seconds() / 60
        tbr54 = 100 * float((g < 54).mean())
        s = summary.loc[stem]
        rec_rows.append({
            "recording_id": rec_id, "patient_id": pid, "source_file": path.name,
            "start_date": pd.to_datetime(stem.rsplit("_", 1)[1]).strftime("%Y-%m-%d"),
            "cgm_start": ts(cgm["time"].iat[0]), "cgm_end": ts(cgm["time"].iat[-1]), "n_cgm": len(cgm),
            "completeness_pct": round(100 * len(cgm) / (span / 15 + 1), 2), "max_gap_min": float(dt.max()),
            "quality_flag": "suspected_sensor_bias" if tbr54 > SENSOR_BIAS_TBR54_PCT else None,
            **{c: (None if pd.isna(s[c]) else s[c]) for c in list(SUMMARY_COLUMNS.values()) + ["hba1c_pct"]},
        })

    rec = pd.DataFrame(rec_rows)
    patients = rec.groupby("patient_id").agg(sex=("sex", "first"), n_recordings=("recording_id", "count")).reset_index()
    patients.insert(1, "source_number", patients["patient_id"].str[2:])

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    patients.to_sql("dm_patients", con, if_exists="append", index=False)
    rec.to_sql("dm_recordings", con, if_exists="append", index=False)
    con.executemany("INSERT INTO dm_cgm VALUES (?,?,?,?)", cgm_rows)
    con.executemany("INSERT INTO dm_cbg VALUES (?,?,?,?)", cbg_rows)
    con.executemany("INSERT INTO dm_meals VALUES (?,?,?,?,?,?,?)", meal_rows)
    con.executemany("INSERT INTO dm_medications VALUES (?,?,?,?,?,?,?,?,?)", med_rows)
    con.commit()
    for t in ("dm_patients", "dm_recordings", "dm_cgm", "dm_cbg", "dm_meals", "dm_medications"):
        print(f"{t:<16} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>7}")
    print("flagged:", [r[0] for r in con.execute("SELECT recording_id FROM dm_recordings WHERE quality_flag IS NOT NULL")])
    con.close()
    print(f"wrote {db_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    a = ap.parse_args()
    build(a.raw.expanduser(), a.db)
