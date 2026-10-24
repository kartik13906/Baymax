"""
merge_adapter.py - Baymax Phase 2, step 6: merge the LoRA adapter into the base model.

Ollama and llama.cpp need ONE complete model, not base + adapter. This script
adds the adapter's small corrections into the base weights and saves the result
as a normal Hugging Face model folder, ready for GGUF conversion.

Run:  python merge_adapter.py outputs/baymax-lora-v3
Out:  outputs/baymax-merged/
"""

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "outputs/baymax-lora-v3"
OUT_DIR = "outputs/baymax-merged"

print(f"Merging {ADAPTER} into {BASE_MODEL} ...")

# Load the base in FULL precision (not 4-bit). Merging into 4-bit weights would
# lose precision. CPU is fine here (the merge is just additions) and keeps the
# GPU free.
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map="cpu"
)
model = PeftModel.from_pretrained(base, ADAPTER)
merged = model.merge_and_unload()          # folds the LoRA matrices into the weights

merged.save_pretrained(OUT_DIR, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(OUT_DIR)
print(f"Saved merged model to {OUT_DIR}")
