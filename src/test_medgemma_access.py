from transformers import AutoProcessor

MODEL_ID = "google/medgemma-1.5-4b-it"

print("Loading MedGemma processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("MedGemma access successful.")
print("Processor:", type(processor).__name__)