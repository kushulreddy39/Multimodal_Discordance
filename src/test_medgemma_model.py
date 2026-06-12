import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "google/medgemma-1.5-4b-it"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    device_name = torch.cuda.get_device_name(0)
    total_memory_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )

    print("=" * 60)
    print("Loading MedGemma 1.5 on GPU")
    print("=" * 60)
    print("Model:", MODEL_ID)
    print("GPU:", device_name)
    print("GPU memory:", f"{total_memory_gb:.2f} GB")
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    start_time = time.time()

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    print("Processor loaded.")
    print("\nDownloading/loading model weights...")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,

        # Approximately half the memory of float32.
        dtype=torch.bfloat16,

        # Place as much of the model as possible on the GPU.
        device_map="auto",

        # Leave some GPU memory available for inference.
        max_memory={
            0: "14GiB",
            "cpu": "32GiB",
        },

        low_cpu_mem_usage=True,
    )

    model.eval()

    elapsed_minutes = (
        time.time() - start_time
    ) / 60

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    allocated_gb = (
        torch.cuda.memory_allocated(0) / 1024**3
    )

    reserved_gb = (
        torch.cuda.memory_reserved(0) / 1024**3
    )

    print("\n" + "=" * 60)
    print("MEDGEMMA LOADED SUCCESSFULLY")
    print("=" * 60)
    print("Processor:", type(processor).__name__)
    print("Model:", type(model).__name__)
    print("Parameters:", f"{parameter_count:,}")
    print("Loading time:", f"{elapsed_minutes:.2f} minutes")
    print("GPU memory allocated:", f"{allocated_gb:.2f} GB")
    print("GPU memory reserved:", f"{reserved_gb:.2f} GB")

    if hasattr(model, "hf_device_map"):
        print("Device map:", model.hf_device_map)
    else:
        print("Primary device:", next(model.parameters()).device)


if __name__ == "__main__":
    main()
