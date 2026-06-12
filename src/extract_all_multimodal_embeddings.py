#!/usr/bin/env python3
"""
Extract ECG, CXR, note, and joint CXR+note representations for every valid row.

The script is restartable and runs in two GPU-safe phases:

1. ECG phase
   ECG -> HuBERT-ECG -> 768-dimensional vector

2. MedGemma phase
   CXR -> MedGemma image representation -> 2560 dimensions
   Note -> MedGemma text representation -> 2560 dimensions
   CXR + note -> MedGemma joint representation -> 2560 dimensions

Models are loaded only once per phase. Each record is saved immediately, so an
interrupted run can be restarted without repeating completed records.

Default input:
    data/final_multimodal_dataset.csv

Default output:
    outputs/all_embeddings/

Examples:
    # Small validation run
    python src/extract_all_multimodal_embeddings.py --limit 10

    # Entire dataset
    python src/extract_all_multimodal_embeddings.py

    # Run one phase only
    python src/extract_all_multimodal_embeddings.py --phase ecg
    python src/extract_all_multimodal_embeddings.py --phase medgemma
    python src/extract_all_multimodal_embeddings.py --phase aggregate

    # Rebuild final stacked arrays without model inference
    python src/extract_all_multimodal_embeddings.py --phase aggregate
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pydicom
import torch
import wfdb
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from scipy.signal import resample_poly
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor


ECG_MODEL_ID = "Edoardo-BS/hubert-ecg-base"
MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"

ECG_S3_ROOT = (
    "s3://arn:aws:s3:us-east-1:724665945834:"
    "accesspoint/mimic-iv-ecg-v1-0-01/mimic-iv-ecg/1.0"
)
CXR_S3_ROOT = (
    "s3://arn:aws:s3:us-east-1:724665945834:"
    "accesspoint/mimic-cxr-v2-1-0-01/mimic-cxr/2.1.0"
)

STANDARD_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

TARGET_FS = 500
WINDOW_SECONDS = 5
TARGET_SAMPLES = TARGET_FS * WINDOW_SECONDS
MAX_NOTE_CHARS = 30000
MAX_TOKENS = 4096

ECG_DIM = 768
MEDGEMMA_DIM = 2560


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all multimodal representations with checkpointing."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/final_multimodal_dataset.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/all_embeddings"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "ecg", "medgemma", "aggregate"),
        default="all",
        help="Pipeline phase to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of valid rows to process; 0 means all rows.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this zero-based position within the valid-row list.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even when output files already exist.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any, name: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Missing required value: {name}")
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError(f"Missing required value: {name}")
    return text


def clean_id(value: Any, name: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Missing required ID: {name}")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:120] or "record"


def resolve_subject_id(row: pd.Series) -> str:
    for col in ("cxr_subject_id", "subject_id", "ecg_subject_id"):
        if col in row.index and pd.notna(row[col]):
            return clean_id(row[col], col)
    raise ValueError(
        "No subject ID found. Expected cxr_subject_id, subject_id, "
        "or ecg_subject_id."
    )


def resolve_record_id(row: pd.Series, row_index: int) -> str:
    if "record_id" in row.index and pd.notna(row["record_id"]):
        return clean_id(row["record_id"], "record_id")
    return f"ROW_{row_index:06d}"


def resolve_report_path(
    row: pd.Series,
    subject_id: str,
    study_id: str,
) -> str:
    for col in (
        "report_path",
        "note_path",
        "radiology_report_path",
        "cxr_report_path",
    ):
        if col in row.index and pd.notna(row[col]):
            return clean_text(row[col], col)

    return f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}.txt"


def record_key(row: pd.Series, row_index: int) -> str:
    rid = resolve_record_id(row, row_index)
    return f"{row_index:06d}_{safe_name(rid)}"


def join_s3(root: str, relative: str) -> str:
    return root.rstrip("/") + "/" + relative.lstrip("/")


def aws_copy(s3_uri: str, local_path: Path, retries: int = 3) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""

    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "aws", "s3", "cp",
                s3_uri,
                str(local_path),
                "--only-show-errors",
                "--region", "us-east-1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return

        last_error = result.stderr.strip() or result.stdout.strip()
        if attempt < retries:
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"AWS download failed after {retries} attempts:\n"
        f"{s3_uri}\n{last_error}"
    )


def append_status(
    path: Path,
    *,
    phase: str,
    row_index: int,
    record_id: str,
    key: str,
    status: str,
    seconds: float,
    error: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_utc",
        "phase",
        "row_index",
        "record_id",
        "record_key",
        "status",
        "seconds",
        "error",
    ]
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": utc_now(),
                "phase": phase,
                "row_index": row_index,
                "record_id": record_id,
                "record_key": key,
                "status": status,
                "seconds": f"{seconds:.3f}",
                "error": error[:4000],
            }
        )


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    os.replace(temp_path, path)


def valid_vector(path: Path, expected_dim: int) -> bool:
    if not path.exists():
        return False
    try:
        vector = np.load(path, mmap_mode="r")
        return (
            vector.shape == (expected_dim,)
            and np.isfinite(vector).all()
        )
    except Exception:
        return False


def normalize_lead(name: str) -> str:
    value = str(name).strip().replace(" ", "")
    aliases = {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
        "avr": "aVR",
        "avl": "aVL",
        "avf": "aVF",
    }
    return aliases.get(value, value)


def reorder_ecg(signal: np.ndarray, names: Iterable[str]) -> np.ndarray:
    normalized = [normalize_lead(x) for x in names]
    mapping = {name: i for i, name in enumerate(normalized)}
    missing = [lead for lead in STANDARD_LEADS if lead not in mapping]

    if missing:
        raise ValueError(
            f"Missing ECG leads: {missing}. Available leads: {normalized}"
        )

    return np.stack(
        [signal[:, mapping[lead]] for lead in STANDARD_LEADS],
        axis=0,
    ).astype(np.float32)


def prepare_hubert_input(
    ecg_12_lead: np.ndarray,
    source_fs: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_fs = int(round(float(source_fs)))
    if source_fs <= 0:
        raise ValueError(f"Invalid ECG sampling rate: {source_fs}")

    if source_fs != TARGET_FS:
        divisor = math.gcd(source_fs, TARGET_FS)
        ecg_12_lead = resample_poly(
            ecg_12_lead,
            up=TARGET_FS // divisor,
            down=source_fs // divisor,
            axis=1,
        ).astype(np.float32)

    ecg_12_lead = np.nan_to_num(
        ecg_12_lead,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    valid = min(ecg_12_lead.shape[1], TARGET_SAMPLES)
    cropped = ecg_12_lead[:, :valid]

    if valid < TARGET_SAMPLES:
        cropped = np.pad(
            cropped,
            ((0, 0), (0, TARGET_SAMPLES - valid)),
            mode="constant",
        )

    flattened = cropped.reshape(-1).astype(np.float32)

    mask = np.zeros((12, TARGET_SAMPLES), dtype=np.int64)
    mask[:, :valid] = 1

    return (
        torch.from_numpy(flattened).unsqueeze(0),
        torch.from_numpy(mask.reshape(-1)).unsqueeze(0),
    )


def dicom_to_rgb(path: Path) -> Image.Image:
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array

    try:
        pixels = apply_voi_lut(pixels, ds)
    except Exception:
        pass

    array = np.asarray(pixels, dtype=np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = array.max() - array

    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("CXR DICOM contains no finite pixels.")

    low, high = np.percentile(array[finite], [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(array))
        high = float(np.nanmax(array))

    array = np.clip(array, low, high)
    array = (array - low) / max(high - low, 1e-8)
    array = (array * 255.0).astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


def masked_mean(
    hidden: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None or mask.shape[1] != hidden.shape[1]:
        return hidden.mean(dim=1)

    mask = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
    return (
        (hidden * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1.0)
    )


def pool_image_states(image_states: torch.Tensor) -> torch.Tensor:
    if image_states.ndim == 4:
        return image_states.mean(dim=(1, 2))
    if image_states.ndim == 3:
        return image_states.mean(dim=1)
    if image_states.ndim == 2:
        return image_states.mean(dim=0, keepdim=True)
    raise ValueError(
        f"Unexpected image-hidden-state shape: {tuple(image_states.shape)}"
    )


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    float_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}

    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=float_dtype)
        else:
            moved[key] = value.to(device=device)

    return moved


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    required = ["ecg_path", "cxr_path", "cxr_study_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    subject_columns = [
        col
        for col in ("cxr_subject_id", "subject_id", "ecg_subject_id")
        if col in df.columns
    ]
    if not subject_columns:
        raise ValueError(
            "Need one subject-ID column: cxr_subject_id, subject_id, "
            "or ecg_subject_id."
        )

    valid_mask = (
        df["ecg_path"].notna()
        & df["cxr_path"].notna()
        & df["cxr_study_id"].notna()
        & df[subject_columns].notna().any(axis=1)
    )

    valid_df = df.loc[valid_mask].copy()
    if valid_df.empty:
        raise ValueError("No row contains all required modality paths and IDs.")

    return valid_df


def select_rows(
    valid_df: pd.DataFrame,
    start_index: int,
    limit: int,
) -> pd.DataFrame:
    if start_index < 0:
        raise ValueError("--start-index must be at least 0.")
    if limit < 0:
        raise ValueError("--limit must be at least 0.")

    selected = valid_df.iloc[start_index:]
    if limit > 0:
        selected = selected.iloc[:limit]
    return selected


def extract_one_ecg(
    row: pd.Series,
    row_index: int,
    *,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    ecg_path = clean_text(row["ecg_path"], "ecg_path")
    ecg_base_relative = ecg_path

    if ecg_base_relative.endswith(".hea") or ecg_base_relative.endswith(".dat"):
        ecg_base_relative = str(Path(ecg_base_relative).with_suffix(""))

    temp_dir = Path(tempfile.mkdtemp(prefix="batch_ecg_"))

    try:
        local_ecg_base = temp_dir / Path(ecg_base_relative).name

        for extension in (".hea", ".dat"):
            aws_copy(
                join_s3(ECG_S3_ROOT, ecg_base_relative + extension),
                Path(str(local_ecg_base) + extension),
            )

        wfdb_record = wfdb.rdrecord(str(local_ecg_base))
        if wfdb_record.p_signal is None:
            raise ValueError("WFDB p_signal is unavailable.")

        ecg = reorder_ecg(
            wfdb_record.p_signal,
            wfdb_record.sig_name,
        )

        input_values, attention_mask = prepare_hubert_input(
            ecg,
            float(wfdb_record.fs),
        )
        input_values = input_values.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        attention_mask = attention_mask.to(
            device=device,
            non_blocking=True,
        )

        with torch.inference_mode():
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True,
            )

        vector = (
            outputs.last_hidden_state
            .mean(dim=1)[0]
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if vector.shape != (ECG_DIM,):
            raise ValueError(
                f"Unexpected ECG vector shape: {vector.shape}; "
                f"expected {(ECG_DIM,)}."
            )
        if not np.isfinite(vector).all():
            raise ValueError("ECG vector contains NaN or Inf.")

        return vector

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_one_medgemma(
    row: pd.Series,
    row_index: int,
    *,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    float_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_id = resolve_subject_id(row)
    study_id = clean_id(row["cxr_study_id"], "cxr_study_id")
    cxr_path = clean_text(row["cxr_path"], "cxr_path")
    report_path = resolve_report_path(row, subject_id, study_id)

    temp_dir = Path(tempfile.mkdtemp(prefix="batch_medgemma_"))

    try:
        cxr_suffix = Path(cxr_path).suffix or ".dcm"
        local_cxr = temp_dir / f"cxr{cxr_suffix}"
        local_report = temp_dir / "report.txt"

        aws_copy(join_s3(CXR_S3_ROOT, cxr_path), local_cxr)
        aws_copy(join_s3(CXR_S3_ROOT, report_path), local_report)

        note_text = local_report.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()[:MAX_NOTE_CHARS]

        if not note_text:
            raise ValueError("Downloaded radiology report is empty.")

        image = dicom_to_rgb(local_cxr)

        text_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Clinical radiology report:\n"
                            f"{note_text}\n\n"
                            "Represent the clinically relevant information."
                        ),
                    }
                ],
            }
        ]

        text_inputs = processor.apply_chat_template(
            text_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
        )
        text_inputs = move_batch(text_inputs, device, float_dtype)

        with torch.inference_mode():
            text_outputs = model(
                **text_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        note_vector = (
            masked_mean(
                text_outputs.hidden_states[-1],
                text_inputs.get("attention_mask"),
            )[0]
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        del text_outputs, text_inputs

        joint_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Clinical radiology report:\n"
                            f"{note_text}\n\n"
                            "Integrate the chest radiograph and report into "
                            "one clinical representation."
                        ),
                    },
                ],
            }
        ]

        joint_inputs = processor.apply_chat_template(
            joint_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            do_pan_and_scan=False,
        )
        joint_inputs = move_batch(joint_inputs, device, float_dtype)

        with torch.inference_mode():
            joint_outputs = model(
                **joint_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        if joint_outputs.image_hidden_states is None:
            raise RuntimeError(
                "MedGemma did not return image_hidden_states."
            )

        cxr_vector = (
            pool_image_states(joint_outputs.image_hidden_states)[0]
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        joint_vector = (
            masked_mean(
                joint_outputs.hidden_states[-1],
                joint_inputs.get("attention_mask"),
            )[0]
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        expected = {
            "CXR": cxr_vector,
            "note": note_vector,
            "joint": joint_vector,
        }
        for name, vector in expected.items():
            if vector.shape != (MEDGEMMA_DIM,):
                raise ValueError(
                    f"Unexpected {name} vector shape: {vector.shape}; "
                    f"expected {(MEDGEMMA_DIM,)}."
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"{name} vector contains NaN or Inf.")

        del joint_outputs, joint_inputs, image
        return cxr_vector, note_vector, joint_vector

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def output_paths(output_dir: Path, key: str) -> dict[str, Path]:
    return {
        "ecg": output_dir / "ecg" / f"{key}.npy",
        "cxr": output_dir / "cxr" / f"{key}.npy",
        "note": output_dir / "note" / f"{key}.npy",
        "joint": output_dir / "joint" / f"{key}.npy",
    }


def run_ecg_phase(
    rows: pd.DataFrame,
    output_dir: Path,
    force: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda:0")
    status_path = output_dir / "logs" / "ecg_status.csv"

    print("\nLoading HuBERT-ECG once...")
    model = AutoModel.from_pretrained(
        ECG_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    total = len(rows)
    success = 0
    skipped = 0
    failed = 0

    try:
        for position, (row_index, row) in enumerate(rows.iterrows(), start=1):
            rid = resolve_record_id(row, int(row_index))
            key = record_key(row, int(row_index))
            path = output_paths(output_dir, key)["ecg"]

            if not force and valid_vector(path, ECG_DIM):
                skipped += 1
                print(
                    f"[ECG {position}/{total}] SKIP {rid} "
                    f"(already complete)"
                )
                continue

            started = time.perf_counter()
            try:
                vector = extract_one_ecg(
                    row,
                    int(row_index),
                    model=model,
                    device=device,
                )
                atomic_save_npy(path, vector)
                elapsed = time.perf_counter() - started
                success += 1

                append_status(
                    status_path,
                    phase="ecg",
                    row_index=int(row_index),
                    record_id=rid,
                    key=key,
                    status="success",
                    seconds=elapsed,
                )
                print(
                    f"[ECG {position}/{total}] OK {rid} "
                    f"{vector.shape} {elapsed:.1f}s"
                )

            except Exception as exc:
                elapsed = time.perf_counter() - started
                failed += 1
                message = f"{type(exc).__name__}: {exc}"

                append_status(
                    status_path,
                    phase="ecg",
                    row_index=int(row_index),
                    record_id=rid,
                    key=key,
                    status="failed",
                    seconds=elapsed,
                    error=message,
                )
                print(
                    f"[ECG {position}/{total}] FAILED {rid}: {message}"
                )

            finally:
                gc.collect()
                torch.cuda.empty_cache()

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\nECG phase complete: success={success}, "
        f"skipped={skipped}, failed={failed}"
    )


def run_medgemma_phase(
    rows: pd.DataFrame,
    output_dir: Path,
    force: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    float_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    status_path = output_dir / "logs" / "medgemma_status.csv"

    print("\nLoading MedGemma processor once...")
    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)

    print("Loading MedGemma once...")
    model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_MODEL_ID,
        torch_dtype=float_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "14GiB", "cpu": "32GiB"},
    )
    model.eval()
    device = model.device

    total = len(rows)
    success = 0
    skipped = 0
    failed = 0

    try:
        for position, (row_index, row) in enumerate(rows.iterrows(), start=1):
            rid = resolve_record_id(row, int(row_index))
            key = record_key(row, int(row_index))
            paths = output_paths(output_dir, key)

            complete = (
                valid_vector(paths["cxr"], MEDGEMMA_DIM)
                and valid_vector(paths["note"], MEDGEMMA_DIM)
                and valid_vector(paths["joint"], MEDGEMMA_DIM)
            )

            if not force and complete:
                skipped += 1
                print(
                    f"[MED {position}/{total}] SKIP {rid} "
                    f"(already complete)"
                )
                continue

            started = time.perf_counter()
            try:
                cxr_vector, note_vector, joint_vector = (
                    extract_one_medgemma(
                        row,
                        int(row_index),
                        model=model,
                        processor=processor,
                        device=device,
                        float_dtype=float_dtype,
                    )
                )

                atomic_save_npy(paths["cxr"], cxr_vector)
                atomic_save_npy(paths["note"], note_vector)
                atomic_save_npy(paths["joint"], joint_vector)

                elapsed = time.perf_counter() - started
                success += 1

                append_status(
                    status_path,
                    phase="medgemma",
                    row_index=int(row_index),
                    record_id=rid,
                    key=key,
                    status="success",
                    seconds=elapsed,
                )
                print(
                    f"[MED {position}/{total}] OK {rid} "
                    f"cxr={cxr_vector.shape} note={note_vector.shape} "
                    f"joint={joint_vector.shape} {elapsed:.1f}s"
                )

            except Exception as exc:
                elapsed = time.perf_counter() - started
                failed += 1
                message = f"{type(exc).__name__}: {exc}"

                append_status(
                    status_path,
                    phase="medgemma",
                    row_index=int(row_index),
                    record_id=rid,
                    key=key,
                    status="failed",
                    seconds=elapsed,
                    error=message,
                )
                print(
                    f"[MED {position}/{total}] FAILED {rid}: {message}"
                )

            finally:
                gc.collect()
                torch.cuda.empty_cache()

    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\nMedGemma phase complete: success={success}, "
        f"skipped={skipped}, failed={failed}"
    )


def aggregate_embeddings(
    rows: pd.DataFrame,
    output_dir: Path,
) -> None:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    complete_records: list[dict[str, Any]] = []

    for row_index, row in rows.iterrows():
        row_index_int = int(row_index)
        rid = resolve_record_id(row, row_index_int)
        key = record_key(row, row_index_int)
        paths = output_paths(output_dir, key)

        if not (
            valid_vector(paths["ecg"], ECG_DIM)
            and valid_vector(paths["cxr"], MEDGEMMA_DIM)
            and valid_vector(paths["note"], MEDGEMMA_DIM)
            and valid_vector(paths["joint"], MEDGEMMA_DIM)
        ):
            continue

        subject_id = resolve_subject_id(row)
        study_id = clean_id(row["cxr_study_id"], "cxr_study_id")

        complete_records.append(
            {
                "row_index": row_index_int,
                "record_id": rid,
                "record_key": key,
                "subject_id": subject_id,
                "cxr_study_id": study_id,
                "ecg_path": clean_text(row["ecg_path"], "ecg_path"),
                "cxr_path": clean_text(row["cxr_path"], "cxr_path"),
                "report_path": resolve_report_path(
                    row,
                    subject_id,
                    study_id,
                ),
            }
        )

    count = len(complete_records)
    print(f"\nComplete records available for aggregation: {count}")

    if count == 0:
        raise RuntimeError(
            "No records have all four valid representation files."
        )

    ecg_memmap = np.lib.format.open_memmap(
        final_dir / "ecg_embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, ECG_DIM),
    )
    cxr_memmap = np.lib.format.open_memmap(
        final_dir / "cxr_representations.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, MEDGEMMA_DIM),
    )
    note_memmap = np.lib.format.open_memmap(
        final_dir / "note_representations.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, MEDGEMMA_DIM),
    )
    joint_memmap = np.lib.format.open_memmap(
        final_dir / "joint_cxr_note_representations.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, MEDGEMMA_DIM),
    )

    for output_index, item in enumerate(complete_records):
        paths = output_paths(output_dir, item["record_key"])
        ecg_memmap[output_index] = np.load(paths["ecg"])
        cxr_memmap[output_index] = np.load(paths["cxr"])
        note_memmap[output_index] = np.load(paths["note"])
        joint_memmap[output_index] = np.load(paths["joint"])

        item["embedding_index"] = output_index

        if (output_index + 1) % 100 == 0 or output_index + 1 == count:
            print(f"Aggregated {output_index + 1}/{count}")

    ecg_memmap.flush()
    cxr_memmap.flush()
    note_memmap.flush()
    joint_memmap.flush()

    del ecg_memmap, cxr_memmap, note_memmap, joint_memmap

    metadata = pd.DataFrame(complete_records)
    metadata = metadata[
        [
            "embedding_index",
            "row_index",
            "record_id",
            "record_key",
            "subject_id",
            "cxr_study_id",
            "ecg_path",
            "cxr_path",
            "report_path",
        ]
    ]
    metadata.to_csv(final_dir / "metadata.csv", index=False)

    manifest = {
        "created_utc": utc_now(),
        "input_records_selected": int(len(rows)),
        "complete_records": count,
        "ecg_model": ECG_MODEL_ID,
        "medgemma_model": MEDGEMMA_MODEL_ID,
        "files": {
            "metadata": "metadata.csv",
            "ecg": {
                "file": "ecg_embeddings.npy",
                "shape": [count, ECG_DIM],
                "dtype": "float32",
            },
            "cxr": {
                "file": "cxr_representations.npy",
                "shape": [count, MEDGEMMA_DIM],
                "dtype": "float32",
            },
            "note": {
                "file": "note_representations.npy",
                "shape": [count, MEDGEMMA_DIM],
                "dtype": "float32",
            },
            "joint": {
                "file": "joint_cxr_note_representations.npy",
                "shape": [count, MEDGEMMA_DIM],
                "dtype": "float32",
            },
        },
    }

    (final_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nFinal arrays saved:")
    print(f"  ECG:   {(count, ECG_DIM)}")
    print(f"  CXR:   {(count, MEDGEMMA_DIM)}")
    print(f"  Note:  {(count, MEDGEMMA_DIM)}")
    print(f"  Joint: {(count, MEDGEMMA_DIM)}")
    print(f"  Directory: {final_dir.resolve()}")


def write_run_configuration(
    args: argparse.Namespace,
    valid_count: int,
    selected_count: int,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "started_utc": utc_now(),
        "csv": str(args.csv),
        "output_dir": str(args.output_dir),
        "phase": args.phase,
        "limit": args.limit,
        "start_index": args.start_index,
        "force": args.force,
        "valid_dataset_records": valid_count,
        "selected_records": selected_count,
        "ecg_model": ECG_MODEL_ID,
        "medgemma_model": MEDGEMMA_MODEL_ID,
        "ecg_dimension": ECG_DIM,
        "cxr_dimension": MEDGEMMA_DIM,
        "note_dimension": MEDGEMMA_DIM,
        "joint_dimension": MEDGEMMA_DIM,
    }
    (args.output_dir / "last_run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if shutil.which("aws") is None:
        raise RuntimeError("AWS CLI is not available in PATH.")

    valid_df = load_dataset(args.csv)
    rows = select_rows(valid_df, args.start_index, args.limit)

    if rows.empty:
        raise ValueError("No rows selected for processing.")

    write_run_configuration(
        args,
        valid_count=len(valid_df),
        selected_count=len(rows),
    )

    print("=" * 78)
    print("ALL-RECORD MULTIMODAL REPRESENTATION EXTRACTION")
    print("=" * 78)
    print("Input CSV:", args.csv.resolve())
    print("Valid rows in dataset:", len(valid_df))
    print("Rows selected for this run:", len(rows))
    print("Phase:", args.phase)
    print("Output directory:", args.output_dir.resolve())
    print("Restart behavior: completed vectors are skipped")
    print("ECG shape per record:", (ECG_DIM,))
    print("CXR shape per record:", (MEDGEMMA_DIM,))
    print("Note shape per record:", (MEDGEMMA_DIM,))
    print("Joint shape per record:", (MEDGEMMA_DIM,))

    started = time.perf_counter()

    try:
        if args.phase in ("all", "ecg"):
            run_ecg_phase(rows, args.output_dir, args.force)

        if args.phase in ("all", "medgemma"):
            run_medgemma_phase(rows, args.output_dir, args.force)

        if args.phase in ("all", "aggregate"):
            aggregate_embeddings(rows, args.output_dir)

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user. Completed record files were preserved. "
            "Run the same command again to resume."
        )
        raise SystemExit(130)

    except Exception:
        print("\nFATAL PIPELINE ERROR")
        traceback.print_exc()
        raise

    elapsed = time.perf_counter() - started
    print(f"\nPipeline finished in {elapsed / 3600:.2f} hours.")


if __name__ == "__main__":
    main()
