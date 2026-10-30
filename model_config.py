"""
model_config.py - which local Ollama model to use, and how long to keep it loaded.

The fine-tuned model is available in more than one size. On a 6 GB GPU the app also
holds Whisper and the voice model, so a full Q8 model may not fit and Ollama silently
falls back to the CPU (that alone made the first token take seconds). The Q4 build is
small enough to stay on the GPU and is much faster.

Set BAYMAX_MODEL to force a specific tag. If it is missing, the fallback is used, so
the program still starts on a machine that only has the original `baymax` model.
"""

import os

PREFERRED_MODEL = os.environ.get("BAYMAX_MODEL", "baymax:q4")
FALLBACK_MODEL = os.environ.get("BAYMAX_FALLBACK_MODEL", "baymax")

# Keep the weights (and the prompt cache) in memory between turns. Ollama's default is
# 5 minutes; a cold reload costs seconds and is exactly what a care conversation cannot
# afford after a pause.
KEEP_ALIVE = os.environ.get("BAYMAX_KEEP_ALIVE", "30m")

_resolved = None


def _installed_models():
    """Names Ollama knows about, or None when the server cannot be reached."""
    try:
        import ollama

        response = ollama.list()
    except Exception:
        return None
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    names = set()
    for model in models or []:
        name = getattr(model, "model", None)
        if name is None and isinstance(model, dict):
            name = model.get("model") or model.get("name")
        if not name:
            continue
        names.add(name)
        if name.endswith(":latest"):
            names.add(name[: -len(":latest")])
    return names


def resolve_model():
    """Preferred model if it is installed, otherwise the original one."""
    global _resolved
    if _resolved:
        return _resolved
    names = _installed_models()
    if names is None:
        return PREFERRED_MODEL                    # server will report the real error
    for candidate in (PREFERRED_MODEL, FALLBACK_MODEL):
        if candidate in names or f"{candidate}:latest" in names:
            _resolved = candidate
            return candidate
    _resolved = PREFERRED_MODEL
    return _resolved
