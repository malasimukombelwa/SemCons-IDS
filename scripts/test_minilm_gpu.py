import json
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftModel

run = Path(
    "/project/def-missro01/malj30/semcons_ids/models/"
    "iid_stratified__minilm__seed_42"
)

adapter = run / "slm_adapter"
config = json.loads(
    (adapter / "adapter_config.json").read_text()
)

base_name = config["base_model_name_or_path"]

assert torch.cuda.is_available(), "Aucun GPU CUDA visible"

print("GPU :", torch.cuda.get_device_name(0))
print("PyTorch :", torch.__version__)
print("Modèle :", base_name)

tokenizer = AutoTokenizer.from_pretrained(
    adapter,
    local_files_only=True,
)

base = AutoModelForSequenceClassification.from_pretrained(
    base_name,
    num_labels=10,
    local_files_only=True,
)

model = PeftModel.from_pretrained(
    base,
    str(adapter),
    is_trainable=False,
)
model = model.to("cuda").eval()

print("Nombre de classes :", model.config.num_labels)

# Vérification explicite des poids de classification.
classifier = model.base_model.model.classifier
if hasattr(classifier, "modules_to_save"):
    print("Tête de classification LoRA restaurée")
else:
    print("Tête de classification :", type(classifier).__name__)

texts = [
    "flow_duration: 1000; total_fwd_packets: 2",
    "flow_duration: 50000; total_fwd_packets: 150",
]

inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
).to("cuda")

with torch.inference_mode():
    logits = model(**inputs).logits
    probs = torch.softmax(logits.float(), dim=-1)

assert logits.shape == (2, 10), logits.shape
assert torch.isfinite(probs).all()
assert torch.allclose(
    probs.sum(dim=1),
    torch.ones(2, device="cuda"),
    atol=1e-5,
)

print("Logits :", tuple(logits.shape))
print("Probabilités :", tuple(probs.shape))
print("Classes prédites :", probs.argmax(dim=1).tolist())
print("Sommes :", probs.sum(dim=1).tolist())
print("===== TEST GPU MINILM RÉUSSI =====")
