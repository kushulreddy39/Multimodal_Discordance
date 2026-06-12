#!/usr/bin/env python3
"""
Initial AP-only demographic analysis.

Input:
    data/final_multimodal_dataset.csv

Outputs:
    results/fairness/initial_demographic_analysis/*.csv
    results/fairness/initial_demographic_analysis/initial_demographic_report.md

Only aggregate results are written. No patient-level table is exported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "record_id",
    "ecg_subject_id",
    "cxr_ViewPosition",
    "gender",
    "anchor_age",
    "race",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/final_multimodal_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fairness/initial_demographic_analysis"),
    )
    return parser.parse_args()


def summary_table(series: pd.Series, label: str, denominator: int | None = None) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    denominator = len(series) if denominator is None else denominator

    values = [
        "<MISSING>" if pd.isna(value) else str(value)
        for value in counts.index
    ]

    result = pd.DataFrame(
        {
            label: values,
            "count": counts.values.astype(int),
        }
    )
    result["percentage"] = (result["count"] / denominator * 100).round(2)
    return result


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = " | ".join(df.columns.astype(str))
        separator = " | ".join(["---"] * len(df.columns))
        rows = [
            " | ".join(str(value) for value in row)
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([headers, separator, *rows])


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv.resolve()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean fields.
    df["view_clean"] = (
        df["cxr_ViewPosition"].astype("string").str.strip().str.upper()
    )
    df["race_clean"] = (
        df["race"].astype("string").str.strip().str.upper()
    )

    df["race_group"] = np.select(
        [
            df["race_clean"].str.startswith("BLACK", na=False),
            df["race_clean"].str.startswith("WHITE", na=False),
        ],
        ["Black", "White"],
        default="Other/Excluded",
    )

    df["sex_group"] = df["gender"].map({"F": "Female", "M": "Male"})

    age = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["age_group"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[age < 65, "age_group"] = "Below 65"
    df.loc[age >= 65, "age_group"] = "65 and above"

    # Overall dataset summaries.
    dataset_overview = pd.DataFrame(
        {
            "metric": [
                "total_records",
                "unique_ecg_subjects",
                "missing_view_position",
                "missing_gender",
                "missing_anchor_age",
                "missing_race",
            ],
            "value": [
                len(df),
                df["ecg_subject_id"].nunique(dropna=True),
                int(df["cxr_ViewPosition"].isna().sum()),
                int(df["gender"].isna().sum()),
                int(df["anchor_age"].isna().sum()),
                int(df["race"].isna().sum()),
            ],
        }
    )

    view_counts = summary_table(
        df["view_clean"],
        "cxr_view_position",
    )
    original_race_counts = summary_table(
        df["race_clean"],
        "recorded_race_label",
    )

    # AP-only cohort.
    ap_df = df[df["view_clean"] == "AP"].copy()

    ap_bw_mask = ap_df["race_group"].isin(["Black", "White"])
    ap_bw_df = ap_df[
        ap_bw_mask
        & ap_df["sex_group"].notna()
        & ap_df["age_group"].notna()
    ].copy()

    ap_overview = pd.DataFrame(
        {
            "metric": [
                "ap_records",
                "ap_unique_ecg_subjects",
                "ap_black_white_records",
                "ap_black_white_unique_ecg_subjects",
                "ap_other_or_excluded_race_records",
            ],
            "value": [
                len(ap_df),
                ap_df["ecg_subject_id"].nunique(dropna=True),
                int(ap_bw_mask.sum()),
                ap_df.loc[ap_bw_mask, "ecg_subject_id"].nunique(dropna=True),
                int((ap_df["race_group"] == "Other/Excluded").sum()),
            ],
        }
    )

    ap_race_counts = summary_table(
        ap_df.loc[ap_bw_mask, "race_group"],
        "race_group",
        denominator=int(ap_bw_mask.sum()),
    )
    ap_age_counts = summary_table(
        ap_df["age_group"],
        "age_group",
        denominator=len(ap_df),
    )
    ap_gender_counts = summary_table(
        ap_df["sex_group"],
        "sex_group",
        denominator=len(ap_df),
    )

    record_intersection = (
        ap_bw_df.groupby(
            ["race_group", "age_group", "sex_group"],
            dropna=False,
        )
        .size()
        .reset_index(name="record_count")
        .sort_values(["race_group", "age_group", "sex_group"])
        .reset_index(drop=True)
    )
    record_intersection["percentage_of_ap_black_white_records"] = (
        record_intersection["record_count"] / len(ap_bw_df) * 100
    ).round(2)

    patient_intersection = (
        ap_bw_df.groupby(
            ["race_group", "age_group", "sex_group"],
            dropna=False,
        )["ecg_subject_id"]
        .nunique()
        .reset_index(name="unique_patient_count")
        .sort_values(["race_group", "age_group", "sex_group"])
        .reset_index(drop=True)
    )

    unique_patients_by_race = (
        ap_bw_df.groupby("race_group")["ecg_subject_id"]
        .nunique()
        .reset_index(name="unique_patient_count")
    )
    unique_patients_by_age = (
        ap_bw_df.groupby("age_group")["ecg_subject_id"]
        .nunique()
        .reset_index(name="unique_patient_count")
    )
    unique_patients_by_sex = (
        ap_bw_df.groupby("sex_group")["ecg_subject_id"]
        .nunique()
        .reset_index(name="unique_patient_count")
    )

    records_per_patient = (
        ap_bw_df.groupby("ecg_subject_id")
        .size()
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .rename_axis("statistic")
        .reset_index(name="records_per_patient")
    )

    tables = {
        "dataset_overview.csv": dataset_overview,
        "view_position_counts.csv": view_counts,
        "all_original_race_counts.csv": original_race_counts,
        "ap_overview.csv": ap_overview,
        "ap_black_white_race_counts.csv": ap_race_counts,
        "ap_age_counts.csv": ap_age_counts,
        "ap_gender_counts.csv": ap_gender_counts,
        "ap_bw_intersection_record_counts.csv": record_intersection,
        "ap_bw_intersection_unique_patient_counts.csv": patient_intersection,
        "ap_bw_unique_patients_by_race.csv": unique_patients_by_race,
        "ap_bw_unique_patients_by_age.csv": unique_patients_by_age,
        "ap_bw_unique_patients_by_sex.csv": unique_patients_by_sex,
        "ap_bw_records_per_patient_summary.csv": records_per_patient,
    }

    for filename, table in tables.items():
        table.to_csv(args.output_dir / filename, index=False)

    report = f"""# Initial AP-Only Demographic Analysis

## Cohort definitions

- Chest X-ray view: **AP only**
- Recorded race: labels beginning with **BLACK** or **WHITE**
- Administrative sex: **Female** or **Male**
- Age groups: **Below 65** and **65 and above**, based on `anchor_age`
- Patient identifier: `ecg_subject_id`

## Dataset overview

{markdown_table(dataset_overview)}

## Chest X-ray view-position counts

{markdown_table(view_counts)}

## AP-only cohort overview

{markdown_table(ap_overview)}

## AP-only recorded race counts

{markdown_table(ap_race_counts)}

## AP-only age counts

{markdown_table(ap_age_counts)}

## AP-only administrative sex counts

{markdown_table(ap_gender_counts)}

## AP Black/White intersectional record counts

{markdown_table(record_intersection)}

## AP Black/White intersectional unique-patient counts

{markdown_table(patient_intersection)}

## Unique patients by recorded race

{markdown_table(unique_patients_by_race)}

## Unique patients by age group

{markdown_table(unique_patients_by_age)}

## Unique patients by administrative sex

{markdown_table(unique_patients_by_sex)}

## Records per patient

{markdown_table(records_per_patient)}

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
"""

    report_path = args.output_dir / "initial_demographic_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("=" * 72)
    print("INITIAL DEMOGRAPHIC ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Input CSV: {args.csv.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Total records: {len(df)}")
    print(f"Total AP records: {len(ap_df)}")
    print(f"AP Black/White records: {len(ap_bw_df)}")
    print(
        "AP Black/White unique patients:",
        ap_bw_df["ecg_subject_id"].nunique(),
    )

    print("\nAP race counts:")
    print(ap_df.loc[ap_bw_mask, "race_group"].value_counts())

    print("\nAP age counts:")
    print(ap_df["age_group"].value_counts(dropna=False))

    print("\nAP sex counts:")
    print(ap_df["sex_group"].value_counts(dropna=False))

    print("\nIntersectional record counts:")
    print(record_intersection.to_string(index=False))

    print("\nIntersectional unique-patient counts:")
    print(patient_intersection.to_string(index=False))

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
