# Initial AP-Only Demographic Analysis

## Cohort definitions

- Chest X-ray view: **AP only**
- Recorded race: labels beginning with **BLACK** or **WHITE**
- Administrative sex: **Female** or **Male**
- Age groups: **Below 65** and **65 and above**, based on `anchor_age`
- Patient identifier: `ecg_subject_id`

## Dataset overview

metric | value
--- | ---
total_records | 4424
unique_ecg_subjects | 822
missing_view_position | 43
missing_gender | 0
missing_anchor_age | 0
missing_race | 0

## Chest X-ray view-position counts

cxr_view_position | count | percentage
--- | --- | ---
AP | 2478 | 56.01
LATERAL | 977 | 22.08
PA | 784 | 17.72
LL | 142 | 3.21
<MISSING> | 43 | 0.97

## AP-only cohort overview

metric | value
--- | ---
ap_records | 2478
ap_unique_ecg_subjects | 642
ap_black_white_records | 2181
ap_black_white_unique_ecg_subjects | 570
ap_other_or_excluded_race_records | 297

## AP-only recorded race counts

race_group | count | percentage
--- | --- | ---
White | 1719 | 78.82
Black | 462 | 21.18

## AP-only age counts

age_group | count | percentage
--- | --- | ---
65 and above | 1541 | 62.19
Below 65 | 937 | 37.81

## AP-only administrative sex counts

sex_group | count | percentage
--- | --- | ---
Male | 1413 | 57.02
Female | 1065 | 42.98

## AP Black/White intersectional record counts

race_group | age_group | sex_group | record_count | percentage_of_ap_black_white_records
--- | --- | --- | --- | ---
Black | 65 and above | Female | 178 | 8.16
Black | 65 and above | Male | 77 | 3.53
Black | Below 65 | Female | 106 | 4.86
Black | Below 65 | Male | 101 | 4.63
White | 65 and above | Female | 508 | 23.29
White | 65 and above | Male | 643 | 29.48
White | Below 65 | Female | 166 | 7.61
White | Below 65 | Male | 402 | 18.43

## AP Black/White intersectional unique-patient counts

race_group | age_group | sex_group | unique_patient_count
--- | --- | --- | ---
Black | 65 and above | Female | 33
Black | 65 and above | Male | 20
Black | Below 65 | Female | 27
Black | Below 65 | Male | 29
White | 65 and above | Female | 147
White | 65 and above | Male | 177
White | Below 65 | Female | 46
White | Below 65 | Male | 91

## Unique patients by recorded race

race_group | unique_patient_count
--- | ---
Black | 109
White | 461

## Unique patients by age group

age_group | unique_patient_count
--- | ---
65 and above | 377
Below 65 | 193

## Unique patients by administrative sex

sex_group | unique_patient_count
--- | ---
Female | 253
Male | 317

## Records per patient

statistic | records_per_patient
--- | ---
count | 570.0
mean | 3.8263157894736843
std | 3.6145102182768736
min | 1.0
25% | 1.0
50% | 3.0
75% | 5.0
90% | 9.0
95% | 11.0
99% | 18.0
max | 29.0

## Interpretation

The AP-only cohort supports primary analyses comparing recorded Black versus
recorded White patients, female versus male patients, and patients below 65
versus patients 65 and above.

The race × age × sex analysis should be treated as exploratory because some
intersectional groups contain relatively few unique patients. All later model
splits and resampling procedures must be performed at the patient level using
`ecg_subject_id`.

## Data governance

Only aggregate tables are saved by this script. The matched patient-level CSV,
clinical data, and generated patient embeddings must remain excluded from Git.
