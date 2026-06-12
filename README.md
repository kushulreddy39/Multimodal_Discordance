# Multimodal Discordance Detection

This repository contains a research pipeline for extracting and comparing representations from three clinical modalities:

- **ECG**
- **Chest X-ray**
- **Clinical radiology report**

The long-term goal is to detect cases where the modalities are not clinically concordant, such as when the ECG, chest X-ray, and report provide conflicting evidence about the same patient or clinical state.

---

## Project Overview

The current pipeline uses:

- **HuBERT-ECG** for ECG representation extraction
- **MedGemma 1.5 4B** for chest X-ray and clinical-text representations
- **MIMIC-IV-ECG v1.0**
- **MIMIC-CXR v2.1.0**
- A matched multimodal CSV containing ECG paths, CXR paths, subject IDs, and study IDs

For every valid record, the pipeline extracts:

| Modality | Model | Output dimension |
|---|---|---:|
| ECG | `Edoardo-BS/hubert-ecg-base` | 768 |
| Chest X-ray | `google/medgemma-1.5-4b-it` | 2560 |
| Clinical report | `google/medgemma-1.5-4b-it` | 2560 |
| Joint CXR + report | `google/medgemma-1.5-4b-it` | 2560 |

> The MedGemma vectors are pooled hidden-state representations used for research experiments. They should not be described as dedicated contrastive embeddings.

---

## Current Status

Completed:

- [x] GitHub repository configured with SSH access
- [x] Conda environment created on the UNF GPU server
- [x] PyTorch GPU support verified
- [x] NVIDIA A2 GPU verified
- [x] Hugging Face access configured
- [x] MedGemma successfully loaded on GPU
- [x] HuBERT-ECG successfully loaded
- [x] AWS CLI installed under the user account
- [x] MIMIC-CXR access verified
- [x] MIMIC-IV-ECG access verified
- [x] One-patient multimodal extraction completed
- [x] Ten-record batch extraction completed
- [x] Full-dataset extraction launched with checkpointing and restart support

Ten-record validation results:

```text
ECG:   (10, 768)
CXR:   (10, 2560)
Note:  (10, 2560)
Joint: (10, 2560)
```

All ten records completed successfully with no failed records.

---

## Repository Structure

```text
Multimodal_Discordance/
├── data/
│   └── final_multimodal_dataset.csv
├── logs/
│   └── extract_all_embeddings.log
├── outputs/
│   ├── one_patient_test/
│   └── all_embeddings/
├── src/
│   ├── inspect_one_ecg.py
│   ├── test_ecg_model.py
│   ├── test_medgemma_access.py
│   ├── test_medgemma_model.py
│   ├── test_one_multimodal_patient.py
│   └── extract_all_multimodal_embeddings.py
├── .gitignore
├── requirements.txt
└── README.md
```

The `data/`, `outputs/`, and credential files should not be committed to GitHub.

---

## Tested Environment

The pipeline has been tested with:

```text
Operating system: RHEL-based Linux GPU server
Python: 3.12
PyTorch: 2.12.0+cu126
Transformers: 5.11.0
GPU: NVIDIA A2
GPU memory: approximately 15 GB
CUDA available: True
BF16 supported: True
```

---

## Environment Setup

Activate the project environment:

```bash
conda activate multimodal
cd ~/projects/Multimodal_Discordance
```

Install required Python packages when needed:

```bash
python -m pip install \
  numpy \
  pandas \
  scipy \
  wfdb \
  pydicom \
  pillow \
  torch \
  transformers \
  accelerate \
  huggingface_hub
```

Confirm GPU access:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
```

---

## Hugging Face Setup

The following gated model requires approved Hugging Face access:

```text
google/medgemma-1.5-4b-it
```

Authenticate on the GPU server:

```bash
hf auth login
```

Test access:

```bash
hf download google/medgemma-1.5-4b-it config.json
```

---

## AWS Setup

AWS credentials must be configured locally on the GPU server account.

```bash
aws configure
```

Recommended values:

```text
Default region name: us-east-1
Default output format: json
```

Verify identity:

```bash
aws sts get-caller-identity
```

Test MIMIC-CXR access:

```bash
aws s3 ls \
"s3://arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-cxr-v2-1-0-01/mimic-cxr/2.1.0/" \
--region us-east-1
```

Test MIMIC-IV-ECG access:

```bash
aws s3 ls \
"s3://arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-ecg-v1-0-01/mimic-iv-ecg/1.0/" \
--region us-east-1
```

Never commit AWS access keys or secret keys.

---

## Dataset

Expected input file:

```text
data/final_multimodal_dataset.csv
```

The extraction script expects columns that include:

```text
record_id
ecg_path
cxr_path
cxr_study_id
subject_id or cxr_subject_id or ecg_subject_id
```

Example values:

```text
record_id: REC_000001
subject_id: 19717536
cxr_study_id: 52092504
ecg_path: files/p1971/p19717536/s49179593/49179593
cxr_path: files/p19/p19717536/s52092504/ce48ed3d-83bc26d8-0ded690b-9b9970c9-c1ec6a57.dcm
report_path: files/p19/p19717536/s52092504.txt
```

---

## Test HuBERT-ECG

```bash
python src/test_ecg_model.py
```

A successful test should confirm that the model loaded and that CUDA is being used.

---

## Test MedGemma

```bash
python src/test_medgemma_model.py
```

The tested configuration loads MedGemma in BF16 with GPU memory limits appropriate for the NVIDIA A2.

---

## One-Patient Multimodal Test

Run:

```bash
python src/test_one_multimodal_patient.py
```

The script:

1. Reads the multimodal CSV
2. Selects one complete record
3. Downloads the ECG waveform
4. Downloads the chest X-ray
5. Downloads the radiology report
6. Extracts the four representations
7. Saves all outputs immediately

Expected output shapes:

```text
ECG HuBERT:                  (768,)
CXR MedGemma:               (2560,)
Note MedGemma:              (2560,)
Joint CXR+note MedGemma:    (2560,)
```

Outputs are saved under:

```text
outputs/one_patient_test/
```

Expected files:

```text
selected_record.csv
ecg_hubert_embedding.npy
cxr_medgemma_representation.npy
note_medgemma_representation.npy
joint_cxr_note_medgemma_representation.npy
note.txt
summary.json
```

---

## Ten-Record Validation

Before running the entire dataset, validate the batch pipeline on ten records:

```bash
python src/extract_all_multimodal_embeddings.py --limit 10
```

The pipeline runs in two GPU-safe phases:

### Phase 1: ECG

```text
ECG -> HuBERT-ECG -> 768-dimensional vector
```

### Phase 2: MedGemma

```text
CXR -> 2560-dimensional representation
Report -> 2560-dimensional representation
CXR + report -> 2560-dimensional joint representation
```

HuBERT-ECG and MedGemma are not kept in GPU memory at the same time.

---

## Full-Dataset Extraction

Start the full extraction in the background:

```bash
mkdir -p logs

nohup python -u src/extract_all_multimodal_embeddings.py \
  > logs/extract_all_embeddings.log 2>&1 &
```

Display the process ID:

```bash
echo $!
```

Check whether it is running:

```bash
ps -ef | grep extract_all_multimodal_embeddings.py | grep -v grep
```

Monitor progress:

```bash
tail -f logs/extract_all_embeddings.log
```

Press `Ctrl+C` to stop viewing the log. The background process will continue.

View only the latest log lines:

```bash
tail -n 40 logs/extract_all_embeddings.log
```

Check GPU usage:

```bash
nvidia-smi
```

---

## Checkpointing and Restart Support

Each record is saved immediately after successful processing.

If the server disconnects or the process stops, run the same command again:

```bash
nohup python -u src/extract_all_multimodal_embeddings.py \
  > logs/extract_all_embeddings.log 2>&1 &
```

Already completed vectors are detected and skipped automatically.

Do not use `--force` unless all completed embeddings should be recomputed.

Per-record status logs are saved under:

```text
outputs/all_embeddings/logs/
```

---

## Full Extraction Output

After aggregation, the final directory is:

```text
outputs/all_embeddings/final/
```

Expected files:

```text
metadata.csv
manifest.json
ecg_embeddings.npy
cxr_representations.npy
note_representations.npy
joint_cxr_note_representations.npy
```

Expected final shapes:

```text
ecg_embeddings.npy
(number_of_complete_records, 768)

cxr_representations.npy
(number_of_complete_records, 2560)

note_representations.npy
(number_of_complete_records, 2560)

joint_cxr_note_representations.npy
(number_of_complete_records, 2560)
```

Each row across the four arrays corresponds to the same row in:

```text
metadata.csv
```

Verify the final arrays:

```bash
python - <<'PY'
import numpy as np
import pandas as pd

base = "outputs/all_embeddings/final"

metadata = pd.read_csv(f"{base}/metadata.csv")
ecg = np.load(f"{base}/ecg_embeddings.npy", mmap_mode="r")
cxr = np.load(f"{base}/cxr_representations.npy", mmap_mode="r")
note = np.load(f"{base}/note_representations.npy", mmap_mode="r")
joint = np.load(
    f"{base}/joint_cxr_note_representations.npy",
    mmap_mode="r",
)

print("Metadata:", metadata.shape)
print("ECG:", ecg.shape)
print("CXR:", cxr.shape)
print("Note:", note.shape)
print("Joint:", joint.shape)
PY
```

---

## Representation Notes

The model outputs have different dimensions:

```text
ECG: 768
CXR: 2560
Note: 2560
Joint: 2560
```

Raw ECG and MedGemma vectors should not be compared directly because they come from different representation spaces.

A later modeling stage will project the separate modalities into a shared space, for example:

```text
ECG 768   -> projection network -> 256 dimensions
CXR 2560  -> projection network -> 256 dimensions
Note 2560 -> projection network -> 256 dimensions
```

---

## Planned Discordance Experiments

The next research stages are:

1. Build a shared representation space
2. Create concordant patient examples
3. Create synthetic discordant examples by replacing one modality
4. Train a concordance-versus-discordance classifier
5. Evaluate which modality caused the conflict
6. Measure modality bias
7. Test selective modality shifting
8. Compare patient-level and time-point-level discordance

Potential discordance examples:

```text
Correct ECG + correct CXR + incorrect note
Correct ECG + incorrect CXR + correct note
Incorrect ECG + correct CXR + correct note
Admission CXR + discharge ECG + admission note
```

---

## Data Security

This project uses controlled clinical datasets.

Do not commit:

```text
Raw ECG files
Chest X-rays
Clinical reports
Matched patient CSV files
Generated patient embeddings
AWS credentials
Hugging Face tokens
Temporary downloaded files
```

Recommended `.gitignore` entries:

```gitignore
data/
outputs/
logs/
.env
.aws/
*.npy
*.npz
*.parquet
*.dcm
*.hea
*.dat
```

---

## Research Disclaimer

This repository is for research use only.

The generated representations and future discordance predictions are not intended for direct clinical diagnosis or patient-care decisions without appropriate validation, governance, and clinical review.
