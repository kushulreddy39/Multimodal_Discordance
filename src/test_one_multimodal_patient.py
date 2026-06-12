#!/usr/bin/env python3
"""
One-patient multimodal embedding smoke test.

Inputs:
  data/final_multimodal_dataset.csv

Outputs:
  outputs/one_patient_test/
    selected_record.csv
    ecg_hubert_embedding.npy
    cxr_medgemma_representation.npy
    note_medgemma_representation.npy
    joint_cxr_note_medgemma_representation.npy
    note.txt
    summary.json
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
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


CSV_PATH = Path("data/final_multimodal_dataset.csv")
OUTPUT_DIR = Path("outputs/one_patient_test")

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


def resolve_subject_id(row: pd.Series) -> str:
    for col in ("cxr_subject_id", "subject_id", "ecg_subject_id"):
        if col in row.index and pd.notna(row[col]):
            return clean_id(row[col], col)
    raise ValueError(
        "No subject-ID column found. Expected cxr_subject_id, subject_id, "
        "or ecg_subject_id."
    )


def join_s3(root: str, relative: str) -> str:
    return root.rstrip("/") + "/" + relative.lstrip("/")


def aws_copy(s3_uri: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local_path), "--only-show-errors"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"AWS download failed:\n{s3_uri}\n{result.stderr.strip()}"
        )


def normalize_lead(name: str) -> str:
    value = str(name).strip().replace(" ", "")
    aliases = {
        "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
        "avr": "aVR", "avl": "aVL", "avf": "aVF",
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
        ecg_12_lead, nan=0.0, posinf=0.0, neginf=0.0
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
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


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
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=float_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def print_vector_check(name: str, vector: np.ndarray) -> None:
    print(
        f"{name}: shape={vector.shape}, "
        f"NaN={bool(np.isnan(vector).any())}, "
        f"Inf={bool(np.isinf(vector).any())}, "
        f"L2 norm={float(np.linalg.norm(vector)):.4f}"
    )


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH.resolve()}"
        )

    if shutil.which("aws") is None:
        raise RuntimeError("AWS CLI is not available in PATH.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="one_patient_multimodal_"))

    try:
        df = pd.read_csv(CSV_PATH)

        print("=" * 70)
        print("DATASET INSPECTION")
        print("=" * 70)
        print("Shape:", df.shape)
        print("\nColumns:")
        for col in df.columns:
            print(" -", col)

        required = ["ecg_path", "cxr_path", "cxr_study_id"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")

        subject_columns = [
            c for c in ("cxr_subject_id", "subject_id", "ecg_subject_id")
            if c in df.columns
        ]
        if not subject_columns:
            raise ValueError(
                "Need one of cxr_subject_id, subject_id, or ecg_subject_id."
            )

        valid_mask = (
            df["ecg_path"].notna()
            & df["cxr_path"].notna()
            & df["cxr_study_id"].notna()
        )
        valid_mask &= df[subject_columns].notna().any(axis=1)

        if not valid_mask.any():
            raise ValueError("No row contains all required modality paths/IDs.")

        row = df.loc[valid_mask].iloc[0]
        selected_index = row.name

        record_id = (
            clean_id(row["record_id"], "record_id")
            if "record_id" in row.index and pd.notna(row["record_id"])
            else str(selected_index)
        )
        subject_id = resolve_subject_id(row)
        study_id = clean_id(row["cxr_study_id"], "cxr_study_id")
        ecg_path = clean_text(row["ecg_path"], "ecg_path")
        cxr_path = clean_text(row["cxr_path"], "cxr_path")

        print("\nSelected row index:", selected_index)
        print("record_id:", record_id)
        print("subject_id:", subject_id)
        print("study_id:", study_id)
        print("ecg_path:", ecg_path)
        print("cxr_path:", cxr_path)

        row.to_frame().T.to_csv(
            OUTPUT_DIR / "selected_record.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Download ECG
        # -------------------------------------------------------------
        ecg_base_relative = ecg_path
        if ecg_base_relative.endswith(".hea") or ecg_base_relative.endswith(".dat"):
            ecg_base_relative = str(Path(ecg_base_relative).with_suffix(""))

        local_ecg_base = temp_dir / Path(ecg_base_relative).name
        for ext in (".hea", ".dat"):
            aws_copy(
                join_s3(ECG_S3_ROOT, ecg_base_relative + ext),
                Path(str(local_ecg_base) + ext),
            )

        # -------------------------------------------------------------
        # Download CXR
        # -------------------------------------------------------------
        cxr_suffix = Path(cxr_path).suffix or ".dcm"
        local_cxr = temp_dir / f"cxr{cxr_suffix}"
        aws_copy(join_s3(CXR_S3_ROOT, cxr_path), local_cxr)

        # -------------------------------------------------------------
        # Download report
        # -------------------------------------------------------------
        report_relative = (
            f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}.txt"
        )
        local_report = temp_dir / "report.txt"
        aws_copy(join_s3(CXR_S3_ROOT, report_relative), local_report)

        note_text = local_report.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()[:MAX_NOTE_CHARS]

        if not note_text:
            raise ValueError("Downloaded report is empty.")

        (OUTPUT_DIR / "note.txt").write_text(
            note_text,
            encoding="utf-8",
        )

        # -------------------------------------------------------------
        # Load ECG
        # -------------------------------------------------------------
        wfdb_record = wfdb.rdrecord(str(local_ecg_base))
        if wfdb_record.p_signal is None:
            raise ValueError("WFDB p_signal is unavailable.")

        ecg = reorder_ecg(
            wfdb_record.p_signal,
            wfdb_record.sig_name,
        )
        source_fs = float(wfdb_record.fs)

        print("\nECG shape:", ecg.shape)
        print("ECG sampling rate:", source_fs)
        print("ECG leads:", STANDARD_LEADS)
        print("CXR image size:", dicom_to_rgb(local_cxr).size)
        print("Report characters:", len(note_text))

        # -------------------------------------------------------------
        # Load models
        # -------------------------------------------------------------
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available.")

        device = torch.device("cuda:0")
        med_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print("\nLoading HuBERT-ECG...")
        ecg_model = AutoModel.from_pretrained(
            ECG_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device)
        ecg_model.eval()

        print("Loading MedGemma processor...")
        processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)

        print("Loading MedGemma...")
        medgemma = AutoModelForImageTextToText.from_pretrained(
            MEDGEMMA_MODEL_ID,
            torch_dtype=med_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "14GiB", "cpu": "32GiB"},
        )
        medgemma.eval()
        med_device = medgemma.device

        # -------------------------------------------------------------
        # ECG embedding
        # -------------------------------------------------------------
        input_values, attention_mask = prepare_hubert_input(ecg, source_fs)
        input_values = input_values.to(device=device, dtype=torch.float32)
        attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            ecg_outputs = ecg_model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True,
            )
        ecg_embedding = (
            ecg_outputs.last_hidden_state
            .mean(dim=1)[0]
            .float()
            .cpu()
            .numpy()
        )

        # Free some GPU memory before MedGemma forward passes.
        del ecg_model, ecg_outputs, input_values, attention_mask
        torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Clinical-note representation
        # -------------------------------------------------------------
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
        text_inputs = move_batch(text_inputs, med_device, med_dtype)

        with torch.inference_mode():
            text_outputs = medgemma(
                **text_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        note_embedding = (
            masked_mean(
                text_outputs.hidden_states[-1],
                text_inputs.get("attention_mask"),
            )[0]
            .float()
            .cpu()
            .numpy()
        )

        del text_outputs, text_inputs
        torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # CXR representation and joint CXR+note representation
        # -------------------------------------------------------------
        image = dicom_to_rgb(local_cxr)

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
        joint_inputs = move_batch(joint_inputs, med_device, med_dtype)

        with torch.inference_mode():
            joint_outputs = medgemma(
                **joint_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        if joint_outputs.image_hidden_states is None:
            raise RuntimeError(
                "MedGemma did not return image_hidden_states."
            )

        cxr_embedding = (
            pool_image_states(joint_outputs.image_hidden_states)[0]
            .float()
            .cpu()
            .numpy()
        )

        joint_embedding = (
            masked_mean(
                joint_outputs.hidden_states[-1],
                joint_inputs.get("attention_mask"),
            )[0]
            .float()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------------------
        # Save outputs
        # -------------------------------------------------------------
        np.save(OUTPUT_DIR / "ecg_hubert_embedding.npy", ecg_embedding)
        np.save(
            OUTPUT_DIR / "cxr_medgemma_representation.npy",
            cxr_embedding,
        )
        np.save(
            OUTPUT_DIR / "note_medgemma_representation.npy",
            note_embedding,
        )
        np.save(
            OUTPUT_DIR / "joint_cxr_note_medgemma_representation.npy",
            joint_embedding,
        )

        print("\n" + "=" * 70)
        print("ONE-PATIENT EMBEDDING RESULTS")
        print("=" * 70)
        print_vector_check("ECG HuBERT", ecg_embedding)
        print_vector_check("CXR MedGemma", cxr_embedding)
        print_vector_check("Note MedGemma", note_embedding)
        print_vector_check("Joint CXR+note MedGemma", joint_embedding)

        summary = {
            "record_id": record_id,
            "selected_row_index": int(selected_index),
            "subject_id": subject_id,
            "cxr_study_id": study_id,
            "ecg_path": ecg_path,
            "cxr_path": cxr_path,
            "report_path": report_relative,
            "ecg_model": ECG_MODEL_ID,
            "medgemma_model": MEDGEMMA_MODEL_ID,
            "ecg_embedding_shape": list(ecg_embedding.shape),
            "cxr_representation_shape": list(cxr_embedding.shape),
            "note_representation_shape": list(note_embedding.shape),
            "joint_representation_shape": list(joint_embedding.shape),
            "ecg_has_nan": bool(np.isnan(ecg_embedding).any()),
            "cxr_has_nan": bool(np.isnan(cxr_embedding).any()),
            "note_has_nan": bool(np.isnan(note_embedding).any()),
            "joint_has_nan": bool(np.isnan(joint_embedding).any()),
        }

        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\nSaved outputs to:", OUTPUT_DIR.resolve())
        print("SUCCESS: one patient processed across all four representations.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
