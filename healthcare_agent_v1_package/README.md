# Healthcare Agent Synthetic Dataset v1

This is a fully synthetic dataset for a healthcare-agent prototype.

## Tables
- patients: 500
- encounters: 1,200
- vitals: 4,829
- ed_status: 2,160
- beds: 12,960

## Key relationships built into the synthetic data
- Higher ED occupancy tends to increase waiting time and trolley counts.
- Greater ED pressure tends to increase ward bed occupancy.
- COPD tends to lower SpO2 and increase respiratory rate.
- Fever presentations tend to increase temperature and heart rate.
- Higher-acuity triage levels have higher admission probability.

## Recommended first MAF tools
1. get_patient_profile(patient_id)
2. get_patient_encounters(patient_id, limit=5)
3. get_recent_vitals(patient_id, hours=24)
4. get_ed_status(timestamp=None)
5. get_bed_capacity(ward=None, timestamp=None)

## Example SQL
Latest ED status:
```sql
SELECT * FROM ed_status
ORDER BY timestamp DESC
LIMIT 1;
```

Latest vitals for a patient:
```sql
SELECT * FROM vitals
WHERE patient_id = 'P0001'
ORDER BY timestamp DESC
LIMIT 5;
```

Latest bed status by ward:
```sql
SELECT b.*
FROM beds b
JOIN (
    SELECT ward, MAX(timestamp) AS max_ts
    FROM beds
    GROUP BY ward
) x
ON x.ward = b.ward
AND x.max_ts = b.timestamp
ORDER BY b.ward;
```

## Important
This data is synthetic and for research/software prototyping only. It is not real clinical data and must not be used for patient care.
