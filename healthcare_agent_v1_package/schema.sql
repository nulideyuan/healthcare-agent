PRAGMA foreign_keys = ON;

CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL CHECK(age BETWEEN 18 AND 110),
    sex TEXT NOT NULL,
    height_m REAL NOT NULL,
    weight_kg REAL NOT NULL,
    blood_group TEXT,
    chronic_conditions TEXT
);

CREATE TABLE encounters (
    encounter_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    arrival_time TEXT NOT NULL,
    department TEXT NOT NULL,
    chief_complaint TEXT,
    triage_level INTEGER CHECK(triage_level BETWEEN 1 AND 5),
    disposition TEXT,
    admission_time TEXT,
    discharge_time TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE vitals (
    vital_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    encounter_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    heart_rate_bpm REAL,
    respiratory_rate_bpm REAL,
    spo2_pct REAL,
    temperature_c REAL,
    systolic_bp_mmHg REAL,
    diastolic_bp_mmHg REAL,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
);

CREATE TABLE ed_status (
    status_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL UNIQUE,
    patients_in_ed INTEGER,
    patients_waiting INTEGER,
    patients_on_trolleys INTEGER,
    average_wait_minutes INTEGER,
    longest_wait_minutes INTEGER,
    ambulance_arrivals INTEGER,
    ed_arrivals_last_hour INTEGER,
    ed_occupancy_ratio REAL
);

CREATE TABLE beds (
    bed_status_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    ward TEXT NOT NULL,
    specialty TEXT,
    total_beds INTEGER,
    occupied_beds INTEGER,
    available_beds INTEGER,
    blocked_beds INTEGER,
    occupancy_ratio REAL
);

CREATE INDEX idx_encounters_patient ON encounters(patient_id);
CREATE INDEX idx_vitals_patient_time ON vitals(patient_id, timestamp);
CREATE INDEX idx_vitals_encounter ON vitals(encounter_id);
CREATE INDEX idx_ed_status_time ON ed_status(timestamp);
CREATE INDEX idx_beds_time_ward ON beds(timestamp, ward);
