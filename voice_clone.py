"""
voice_clone.py - resident version of the Voicekit / SV2TTS voice cloner.

The `voice-cloning` package reloads the encoder, the synthesizer (twice) and the vocoder
(twice) on *every* call to speech_generator(). For a chat that means tens of seconds before
the first word. This module loads those three models exactly once, computes the speaker
embedding from the registered sample exactly once (and caches it on disk), and then only runs
the two cheap steps per sentence:

    text -> synthesizer -> mel spectrogram -> vocoder -> waveform

Voice samples live where Voicekit puts them:  ~/.voicekit/voices/<name>/profile.json

Use it through speak.load_voicekit_synth() so Baymax keeps a single synth(text) interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_MODELS_DIR_NAMES = ("BAYMAX_VOICE_MODELS", "VOICE_CLONING_MODELS")


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------
def package_models_dir() -> Path:
    """The saved_models/default folder shipped inside the installed voice_cloning package."""
    import importlib.util

    spec = importlib.util.find_spec("voice_cloning")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The `voice-cloning` package is not installed.")
    return Path(list(spec.submodule_search_locations)[0]) / "saved_models" / "default"


def models_dir() -> Path:
    for env_name in DEFAULT_MODELS_DIR_NAMES:
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser()
    return package_models_dir()


def voice_dir(voice_name: str) -> Path:
    """Resolve a Voicekit voice profile folder without importing the heavy backend."""
    home = Path(os.environ.get("VOICEKIT_HOME", "~/.voicekit")).expanduser()
    candidates = [home / "voices" / voice_name]
    # Voicekit stores a slugified name; try the common normalisations too.
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in voice_name.strip().lower())
    candidates.append(home / "voices" / safe.strip("-"))
    for folder in candidates:
        if (folder / "profile.json").is_file():
            return folder
    have = sorted(p.name for p in (home / "voices").glob("*") if p.is_dir())
    raise FileNotFoundError(
        f"No voice named {voice_name!r}. Existing voices: {', '.join(have) or '(none)'}."
    )


def sample_path(voice_name: str) -> Path:
    folder = voice_dir(voice_name)
    profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    return folder / profile.get("sample", "sample.wav")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_wav_16k(path: Path) -> np.ndarray:
    """Read any audio file and return mono float32 at 16 kHz."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.asarray(wav, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def median_f0(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Rough median fundamental frequency of voiced frames (autocorrelation, no extra deps)."""
    import librosa

    y = np.asarray(audio, np.float32)
    if y.size < sr // 10:
        return 0.0
    f0 = librosa.yin(y, fmin=60, fmax=350, sr=sr, frame_length=1024, hop_length=160)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else 0.0


# ---------------------------------------------------------------------------
# The resident cloner
# ---------------------------------------------------------------------------
class VoiceCloner:
    """Loads the SV2TTS models once and keeps them in memory (GPU when available)."""

    def __init__(self, voice_name: str = "baymax", *, models: Path | None = None,
                 sample: Path | None = None, verbose: bool = True):
        self.voice_name = voice_name
        self.models = Path(models) if models else models_dir()
        self.sample = Path(sample) if sample else None      # override for experiments
        self.verbose = verbose
        self._lock = threading.RLock()
        self._loaded = False
        self._embed: np.ndarray | None = None
        self._sample_path: Path | None = None
        self._encoder = None
        self._synthesizer = None
        self._vocoder = None

    # --- setup --------------------------------------------------------------
    def load(self) -> "VoiceCloner":
        """Load encoder + synthesizer + vocoder and compute the speaker embedding (once)."""
        with self._lock:
            if self._loaded:
                return self
            t0 = time.perf_counter()
            for name in ("encoder.pt", "synthesizer.pt", "vocoder.pt"):
                path = self.models / name
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing voice-cloning model {path}. "
                        "Download encoder.pt, synthesizer.pt and vocoder.pt from "
                        "https://huggingface.co/CorentinJ/SV2TTS into that folder."
                    )

            from voice_cloning.encoder import inference as encoder
            from voice_cloning.synthesizer.inference import Synthesizer
            from voice_cloning.vocoder import inference as vocoder

            encoder.load_model(self.models / "encoder.pt")
            self._vocoder = vocoder
            vocoder.load_model(str(self.models / "vocoder.pt"), verbose=self.verbose)
            self._synthesizer = Synthesizer(self.models / "synthesizer.pt", verbose=self.verbose)
            self._synthesizer.load()                     # build Tacotron now, not on first sentence
            self._encoder = encoder

            self._sample_path = self.sample or sample_path(self.voice_name)
            self._embed = self._load_or_compute_embedding(self._sample_path)
            self._loaded = True
            if self.verbose:
                print(
                    f"[voice-clone] ready in {time.perf_counter() - t0:.1f}s "
                    f"(voice '{self.voice_name}', sample {self._sample_path.name})"
                )
            return self

    def _embedding_cache_path(self, sample: Path) -> Path:
        stat = sample.stat()
        key = hashlib.sha1(
            f"{sample.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
        ).hexdigest()[:16]
        cache = Path(os.environ.get("BAYMAX_VOICE_CACHE", "~/.cache/baymax-voice")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        return cache / f"{self.voice_name}-{key}.npy"

    def _load_or_compute_embedding(self, sample: Path) -> np.ndarray:
        cache = self._embedding_cache_path(sample)
        if cache.is_file():
            embed = np.load(cache)
            if self.verbose:
                print(f"[voice-clone] speaker embedding loaded from cache ({cache.name})")
            return np.asarray(embed, dtype=np.float32)

        t0 = time.perf_counter()
        wav = load_wav_16k(sample)
        wav = self._encoder.preprocess_wav(wav, SAMPLE_RATE)
        embed = np.asarray(self._encoder.embed_utterance(wav), dtype=np.float32)
        np.save(cache, embed)
        if self.verbose:
            print(
                f"[voice-clone] speaker embedding from {len(wav) / SAMPLE_RATE:.0f}s of audio "
                f"in {time.perf_counter() - t0:.1f}s"
            )
        return embed

    # --- inference ----------------------------------------------------------
    @property
    def embedding(self) -> np.ndarray:
        self.load()
        return self._embed

    def synth(self, text: str, embedding: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """text -> (float32 mono audio, 16000 Hz). Models stay loaded between calls.
        `embedding` is only for A/B testing candidate reference clips; normal use reads the
        cached embedding of the registered voice."""
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        self.load()
        embed = self._embed if embedding is None else np.asarray(embedding, dtype=np.float32)
        with self._lock:
            import torch

            with torch.no_grad():                 # inference_mode() breaks this old WaveRNN
                specs = self._synthesizer.synthesize_spectrograms([text], [embed])
                wav = self._vocoder.infer_waveform(
                    specs[0],
                    progress_callback=lambda *a, **k: None,   # no progress bars in a chat
                )
            wav = np.pad(wav, (0, self._synthesizer.sample_rate), mode="constant")
            wav = self._encoder.preprocess_wav(wav)
            return np.asarray(wav, dtype=np.float32), SAMPLE_RATE

    # --- convenience for experiments ---------------------------------------
    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        """Speaker embedding of a waveform, for comparing a clone against the reference."""
        self.load()
        wav = np.asarray(audio, np.float32)
        wav = self._encoder.preprocess_wav(wav, SAMPLE_RATE)
        return np.asarray(self._encoder.embed_utterance(wav), dtype=np.float32)


_singletons: dict[tuple, VoiceCloner] = {}
_singletons_lock = threading.Lock()


def get_cloner(voice_name: str = "baymax", **kwargs) -> VoiceCloner:
    """Process-wide cached cloner so repeated load_voicekit_synth() calls share the models."""
    key = (voice_name, str(kwargs.get("models") or models_dir()), str(kwargs.get("sample") or ""))
    with _singletons_lock:
        if key not in _singletons:
            _singletons[key] = VoiceCloner(voice_name, **kwargs)
        return _singletons[key]
