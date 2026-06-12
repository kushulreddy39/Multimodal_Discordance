import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import wfdb


CSV_FILE = Path(
    r"C:\Users\kushu\OneDrive\Desktop\PRIMED-AI\files"
    r"\final_multimodal_dataset.csv"
)

ECG_S3_ROOT = (
    "s3://arn:aws:s3:us-east-1:"
    "724665945834:"
    "accesspoint/"
    "mimic-iv-ecg-v1-0-01/"
    "mimic-iv-ecg/1.0/"
)


def remove_ecg_extension(ecg_path: str) -> str:
    """Remove .hea or .dat if the CSV path already includes it."""

    ecg_path = ecg_path.strip()

    for extension in (".hea", ".dat"):
        if ecg_path.lower().endswith(extension):
            return ecg_path[: -len(extension)]

    return ecg_path


def main() -> None:
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Dataset was not found:\n{CSV_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    required_columns = {
        "record_id",
        "ecg_subject_id",
        "ecg_path",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing_columns)}"
        )

    valid_rows = df.dropna(subset=["ecg_path"])

    if valid_rows.empty:
        raise ValueError("No valid ECG paths were found.")

    row = valid_rows.iloc[0]

    record_id = row["record_id"]
    subject_id = row["ecg_subject_id"]

    ecg_path = remove_ecg_extension(
        str(row["ecg_path"])
    )

    record_name = PurePosixPath(ecg_path).name

    print("=" * 60)
    print("Inspecting one ECG")
    print("=" * 60)
    print("Record ID:", record_id)
    print("Subject ID:", subject_id)
    print("ECG path:", ecg_path)

    with tempfile.TemporaryDirectory() as temp_directory:
        local_base = Path(temp_directory) / record_name

        for extension in (".hea", ".dat"):
            s3_source = (
                ECG_S3_ROOT
                + ecg_path
                + extension
            )

            local_destination = (
                str(local_base) + extension
            )

            print(
                f"\nDownloading {extension}:",
                s3_source,
            )

            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    s3_source,
                    local_destination,
                ],
                check=True,
            )

        print("\nLoading ECG with WFDB...")

        record = wfdb.rdrecord(
            str(local_base)
        )

        signal = np.asarray(
            record.p_signal,
            dtype=np.float32,
        )

        print("\nECG loaded successfully.")
        print("Signal shape:", signal.shape)
        print("Sampling rate:", record.fs)
        print("Lead names:", record.sig_name)
        print("Duration in seconds:", signal.shape[0] / record.fs)
        print("Contains NaN:", np.isnan(signal).any())
        print("Minimum value:", np.nanmin(signal))
        print("Maximum value:", np.nanmax(signal))


if __name__ == "__main__":
    main()