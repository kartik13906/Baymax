"""
xtts_voice.py - resident XTTS v2 voice cloning for Baymax.

XTTS v2 is a modern zero-shot cloner: it needs only a few seconds of clean reference
audio and keeps the speaker's timbre, pitch and accent far better than the 2019 SV2TTS
model behind Voicekit. It is also much faster here (~0.3x real time on the RTX 3050
instead of ~1.0x for SV2TTS's WaveRNN).

The model is loaded once per process and the conditioning latents (the expensive part of
reading the reference) are cached on disk, so a sentence only runs the GPT + HiFiGAN
decoder.

Reference clips for a voice live in:
    ~/.voicekit/voices/<name>/xtts_refs/*.wav
and fall back to the Voicekit profile sample. Any of those paths can be overridden with
BAYMAX_XTTS_REFS (colon-separated list of wav files).

The clips are used in filename order. Only the first 6 seconds drive the GPT prosody
(and therefore the speaking pace), so keep a naturally paced clip first. The speaker
embedding is averaged over all clips, which keeps the timbre stable. Temperature is
lowered from XTTS's 0.75 default because the same sentence otherwise varied from 3.8
to 4.8 seconds; 0.65 keeps it near 3.2-3.6 seconds and closer to the reference pitch.

Use it through speak.load_xtts_synth() so Baymax keeps synth(text) -> (audio, sr).
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MAX_REF_SECONDS = 30


# ---------------------------------------------------------------------------
# Where the reference clips are
# ---------------------------------------------------------------------------
def _voice_dir(voice_name: str) -> Path:
    from voice_clone import voice_dir

    return voice_dir(voice_name)


def reference_paths(voice_name: str = "baymax") -> list[Path]:
    """Clean reference clips, in priority order. XTTS uses the first 6 s for the GPT
    conditioning and averages the speaker embedding over all of them."""
    env = os.environ.get("BAYMAX_XTTS_REFS")
    if env:
        paths = [Path(p).expanduser() for p in env.split(os.pathsep) if p.strip()]
        if paths:
            return paths

    folder = _voice_dir(voice_name)
    refs = sorted((folder / "xtts_refs").glob("*.wav"))
    if refs:
        return refs

    from voice_clone import sample_path

    return [sample_path(voice_name)]


def refs_fingerprint(voice_name: str = "baymax") -> str:
    """Short hash of the reference clips, so a pre-rendered audio cache is invalidated
    when the clips (or their order) change."""
    h = hashlib.sha1()
    for path in reference_paths(voice_name):
        stat = path.stat()
        h.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}|".encode())
    return h.hexdigest()[:8]


# ---------------------------------------------------------------------------
# The resident cloner
# ---------------------------------------------------------------------------
class XTTSVoice:
    """Loads XTTS v2 once and keeps it (and its reference latents) in memory."""

    def __init__(self, voice_name: str = "baymax", *, refs: list[Path] | None = None,
                 speed: float = 1.0, temperature: float = 0.65, language: str = "en",
                 verbose: bool = True):
        self.voice_name = voice_name
        self.refs = [Path(p) for p in (refs or reference_paths(voice_name))]
        self.speed = float(speed)
        # 0.75 is XTTS's default. Lower values sound more even: at 0.65 the same
        # sentence varied from 3.2 to 3.9 s instead of 3.8 to 4.8 s, and pitch stayed
        # closer to the reference. Too low (0.55) starts to flatten the delivery.
        self.temperature = float(temperature)
        self.language = language
        self.verbose = verbose
        self._lock = threading.RLock()
        self._loaded = False
        self._tts = None
        self._model = None
        self._gpt_cond = None
        self._speaker = None

    # --- setup --------------------------------------------------------------
    def _cache_path(self) -> Path:
        parts = [MODEL_NAME, self.voice_name]
        for p in self.refs:
            st = p.stat()
            parts.append(f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}")
        key = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
        cache = Path(os.environ.get("BAYMAX_VOICE_CACHE", "~/.cache/baymax-voice")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        return cache / f"xtts-{self.voice_name}-{key}.pt"

    def load(self) -> "XTTSVoice":
        with self._lock:
            if self._loaded:
                return self
            import torch
            from TTS.api import TTS

            t0 = time.perf_counter()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tts = TTS(MODEL_NAME, gpu=(device == "cuda"), progress_bar=False)
            self._model = self._tts.synthesizer.tts_model

            cache = self._cache_path()
            loaded = False
            if cache.is_file():
                try:
                    blob = torch.load(cache, map_location="cpu", weights_only=True)
                    self._gpt_cond = blob["gpt_cond"]
                    self._speaker = blob["speaker"]
                    loaded = True
                except Exception:
                    loaded = False
            if not loaded:
                self._gpt_cond, self._speaker = self._model.get_conditioning_latents(
                    audio_path=[str(p) for p in self.refs]
                )
                torch.save(
                    {"gpt_cond": self._gpt_cond.cpu(), "speaker": self._speaker.cpu()},
                    cache,
                )
            self._loaded = True
            if self.verbose:
                print(
                    f"[xtts] ready in {time.perf_counter() - t0:.1f}s on {device} "
                    f"with {len(self.refs)} reference clip(s)"
                )
            return self

    # --- inference ----------------------------------------------------------
    def synth(self, text: str) -> tuple[np.ndarray, int]:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        self.load()
        with self._lock:
            out = self._model.inference(
                text,
                self.language,
                self._gpt_cond,
                self._speaker,
                speed=self.speed,
                temperature=self.temperature,
                enable_text_splitting=True,     # XTTS caps a chunk at ~400 tokens
            )
        wav = np.asarray(out["wav"], dtype=np.float32)
        peak = float(np.abs(wav).max()) if wav.size else 0.0
        if peak > 0.99:
            wav = wav / peak * 0.99
        return wav, SAMPLE_RATE


_singletons: dict[tuple, XTTSVoice] = {}
_singletons_lock = threading.Lock()


def get_xtts(voice_name: str = "baymax", *, speed: float = 1.0, **kwargs) -> XTTSVoice:
    """Process-wide cached XTTS model so repeated loader calls reuse the loaded weights."""
    key = (voice_name, float(speed), float(kwargs.get("temperature", 0.65)),
           tuple(str(p) for p in (kwargs.get("refs") or ())))
    with _singletons_lock:
        if key not in _singletons:
            _singletons[key] = XTTSVoice(voice_name, speed=speed, **kwargs)
        return _singletons[key]
