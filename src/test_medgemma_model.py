import os
import time

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


MODEL_ID = "google/medgemma-1.5-4b-it"

# Optional: hide the Windows symlink warning.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main() -> None:
    print("=" * 60)
    print("Loading MedGemma 1.5 on CPU")
    print("=" * 60)

    print("Device: CPU")
    print("Model:", MODEL_ID)
    print(
        "\nWarning: The first run downloads several GB, "
        "and CPU loading may take a long time."
    )

    start_time = time.time()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    print("\nProcessor loaded.")
    print("Downloading/loading model weights...")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,

        # Float32 is the safest CPU option.
        dtype=torch.float32,

        # Force the complete model onto CPU.
        device_map={"": "cpu"},

        # Avoid creating two complete copies in memory.
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

    print("\n" + "=" * 60)
    print("MEDGEMMA LOADED SUCCESSFULLY")
    print("=" * 60)
    print("Processor:", type(processor).__name__)
    print("Model:", type(model).__name__)
    print("Device:", next(model.parameters()).device)
    print("Parameters:", f"{parameter_count:,}")
    print(
        "Loading time:",
        f"{elapsed_minutes:.2f} minutes"
    )


if __name__ == "__main__":
    main()