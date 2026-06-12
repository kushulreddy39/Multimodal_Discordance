import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
import wfdb

from transformers import AutoModel


# =====================================================
# CONFIGURATION
# =====================================================

CSV_FILE = Path(
    r"C:\Users\kushu\OneDrive\Desktop\PRIMED-AI\files"
    r"\final_multimodal_dataset.csv"
)

OUTPUT_DIR = Path("outputs")

ECG_MODEL_ID = "Edoardo-BS/hubert-ecg-base"

ECG_S3_ROOT = (
    "s3://arn:aws:s3:us-east-1:"
    "724665945834:"
    "accesspoint/"
    "mimic-iv-ecg-v1-0-01/"
    "mimic-iv-ecg/1.0/"
)

TARGET_SAMPLING_RATE = 500
WINDOW_SECONDS = 5
SAMPLES_PER_WINDOW = TARGET_SAMPLING_RATE * WINDOW_SECONDS

STANDARD_12_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


# =====================================================
# HELPERS
# =====================================================

def remove_ecg_extension(ecg_path: str) -> str:
    """Remove .hea or .dat if included in the CSV path."""

    ecg_path = ecg_path.strip()

    for extension in (".hea", ".dat"):
        if ecg_path.lower().endswith(extension):
            return ecg_path[:-len(extension)]

    return ecg_path


def prepare_ecg_windows(record: wfdb.Record) -> torch.Tensor:
    """
    Convert a 10-second, 12-lead ECG into two 5-second model inputs.

    Output shape:
        [2, 30000]

    Explanation:
        12 leads × 2500 samples = 30000 values per window.
    """

    signal = np.asarray(
        record.p_signal,
        dtype=np.float32,
    )

    if signal.ndim != 2:
        raise ValueError(
            f"Expected a 2D ECG signal, received {signal.shape}"
        )

    if int(record.fs) != TARGET_SAMPLING_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLING_RATE} Hz, "
            f"but received {record.fs} Hz."
        )

    available_leads = list(record.sig_name)

    missing_leads = [
        lead
        for lead in STANDARD_12_LEADS
        if lead not in available_leads
    ]

    if missing_leads:
        raise ValueError(
            f"Missing required ECG leads: {missing_leads}"
        )

    # Reorder all records into the same lead order.
    lead_indices = [
        available_leads.index(lead)
        for lead in STANDARD_12_LEADS
    ]

    signal = signal[:, lead_indices]

    # Replace invalid values if any appear in future records.
    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    required_samples = SAMPLES_PER_WINDOW * 2

    if signal.shape[0] < required_samples:
        padding = required_samples - signal.shape[0]

        signal = np.pad(
            signal,
            pad_width=((0, padding), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    # Use the first 10 seconds.
    signal = signal[:required_samples]

    first_window = signal[:SAMPLES_PER_WINDOW]
    second_window = signal[
        SAMPLES_PER_WINDOW:
        required_samples
    ]

    model_inputs = []

    for window in (first_window, second_window):

        # Original shape: [2500 samples, 12 leads]
        # Required organization: [12 leads, 2500 samples]
        window = window.T

        # Flatten lead-by-lead:
        # 12 × 2500 = 30000
        window = window.reshape(-1)

        model_inputs.append(window)

    stacked_inputs = np.stack(
        model_inputs,
        axis=0,
    )

    return torch.tensor(
        stacked_inputs,
        dtype=torch.float32,
    )


def extract_embedding(
    record: wfdb.Record,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Extract one 768-dimensional embedding from a 10-second ECG.

    Each 5-second window is processed separately. The token embeddings
    are averaged within each window, and both window embeddings are then
    averaged to create one ECG-level representation.
    """

    input_values = prepare_ecg_windows(record)

    print("Model input shape:", tuple(input_values.shape))

    input_values = input_values.to(device)

    with torch.inference_mode():
        outputs = model(
            input_values=input_values,
            return_dict=True,
        )

    hidden_states = outputs.last_hidden_state

    print(
        "Model hidden-state shape:",
        tuple(hidden_states.shape),
    )

    # Average model tokens within each 5-second window.
    window_embeddings = hidden_states.mean(dim=1)

    # Average the two 5-second windows.
    ecg_embedding = window_embeddings.mean(dim=0)

    embedding_numpy = (
        ecg_embedding
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    return embedding_numpy, tuple(hidden_states.shape)


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Dataset was not found:\n{CSV_FILE}"
        )

    dataframe = pd.read_csv(CSV_FILE)

    required_columns = {
        "record_id",
        "ecg_subject_id",
        "ecg_path",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    valid_rows = dataframe.dropna(
        subset=["ecg_path"]
    )

    if valid_rows.empty:
        raise ValueError(
            "No valid ECG paths were found."
        )

    # Test only the first valid ECG.
    row = valid_rows.iloc[0]

    record_id = str(row["record_id"])
    subject_id = str(row["ecg_subject_id"])

    ecg_path = remove_ecg_extension(
        str(row["ecg_path"])
    )

    record_name = PurePosixPath(ecg_path).name

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 65)
    print("Extracting one HuBERT-ECG embedding")
    print("=" * 65)
    print("Record ID:", record_id)
    print("Subject ID:", subject_id)
    print("ECG path:", ecg_path)
    print("Device:", device)

    print("\nLoading HuBERT-ECG...")

    model = AutoModel.from_pretrained(
        ECG_MODEL_ID,
        trust_remote_code=True,
        dtype="auto",
    )

    model = model.to(device)
    model.eval()

    print("HuBERT-ECG loaded successfully.")

    with tempfile.TemporaryDirectory() as temp_directory:

        local_base = (
            Path(temp_directory) / record_name
        )

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
                f"\nDownloading {extension} file..."
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

        print("\nLoading ECG...")

        record = wfdb.rdrecord(
            str(local_base)
        )

        print(
            "Original signal shape:",
            record.p_signal.shape,
        )
        print("Sampling rate:", record.fs)
        print("Original lead order:", record.sig_name)
        print("Standardized lead order:", STANDARD_12_LEADS)

        print("\nGenerating embedding...")

        embedding, hidden_shape = extract_embedding(
            record=record,
            model=model,
            device=device,
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            "Embedding contains NaN or infinite values."
        )

    embedding_file = (
        OUTPUT_DIR
        / f"{record_id}_ecg_embedding.npy"
    )

    np.save(
        embedding_file,
        embedding,
    )

    row_output = {
        "record_id": record_id,
        "subject_id": subject_id,
        "ecg_path": ecg_path,
    }

    for index, value in enumerate(embedding):
        row_output[f"emb_{index}"] = float(value)

    parquet_file = (
        OUTPUT_DIR
        / "one_ecg_embedding.parquet"
    )

    pd.DataFrame(
        [row_output]
    ).to_parquet(
        parquet_file,
        index=False,
    )

    print("\n" + "=" * 65)
    print("ECG EMBEDDING EXTRACTION SUCCESSFUL")
    print("=" * 65)
    print("Hidden-state shape:", hidden_shape)
    print("Final embedding shape:", embedding.shape)
    print("Contains NaN:", np.isnan(embedding).any())
    print("Embedding minimum:", embedding.min())
    print("Embedding maximum:", embedding.max())
    print("First 10 values:", embedding[:10])
    print("\nNumPy file:", embedding_file.resolve())
    print("Parquet file:", parquet_file.resolve())


if __name__ == "__main__":
    main()