import torch
from transformers import AutoModel


MODEL_ID = "Edoardo-BS/hubert-ecg-base"


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Testing HuBERT-ECG")
    print("=" * 60)
    print("Device:", device)
    print("Model:", MODEL_ID)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype="auto",
    )

    model = model.to(device)
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nHuBERT-ECG loaded successfully.")
    print("Model class:", type(model).__name__)
    print("Parameters:", f"{parameter_count:,}")


if __name__ == "__main__":
    main()