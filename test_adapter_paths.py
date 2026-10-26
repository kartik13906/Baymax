"""Shared helper: find a LoRA adapter folder (or its newest checkpoint)."""
import glob
import os
import sys


def resolve_adapter(path: str) -> str:
    """Use the adapter folder if it is complete, else the newest checkpoint,
    else stop with a clear message (instead of a confusing Hugging Face 401)."""
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    ckpts = glob.glob(os.path.join(path, "checkpoint-*"))
    if ckpts:
        best = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[1]))
        print(f"No finished adapter in {path}; using newest checkpoint: {best}")
        return best
    sys.exit(f"No adapter found at '{path}'. Run train_qlora.py first, or list the folders with: ls outputs")
