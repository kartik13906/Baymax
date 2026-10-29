"""
speak.py - Baymax Phase 3: text-to-speech with Kokoro-82M (ONNX version).

Two parts:
  1. load_tts()     loads the Kokoro model and returns a function: text -> (audio, sample_rate)
  2. SpeechQueue    two background threads. One turns text into audio, the other plays it.
                    While sentence 1 is playing, sentence 2 is already being synthesized, so
                    there is no gap between sentences. It also measures time-to-first-audio.

Try it:
    python speak.py "Hello, I am Baymax, your personal healthcare companion."
    python speak.py --voice af_heart "I am sorry to hear that."
    python speak.py --list-voices
    python speak.py --save hello.wav "Hello"      # write a file instead of playing (no speaker needed)

The first run downloads the model (about 350 MB) into models/kokoro/.
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.request
import wave

import numpy as np

DEFAULT_MODEL_DIR = os.environ.get("KOKORO_DIR", "models/kokoro")
DEFAULT_VOICE = "am_michael"       # calm male voice. Try af_heart, bm_george, am_adam ...
DEFAULT_SPEED = 0.95               # slightly slower = calmer, more Baymax
BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILE, VOICES_FILE = "kokoro-v1.0.onnx", "voices-v1.0.bin"

DIGIT_WORDS = "zero one two three four five six seven eight nine".split()


def speakable(text):
    """Make text safe to read aloud. Long numbers like 112 would be read as 'one hundred
    twelve'; for phone numbers we want 'one one two'."""
    return re.sub(r"\d{3,}", lambda m: " ".join(DIGIT_WORDS[int(d)] for d in m.group(0)), text)


# ---------------------------------------------------------------
# 1. LOADING THE MODEL
# ---------------------------------------------------------------
def ensure_model_files(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    for name in (MODEL_FILE, VOICES_FILE):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
            continue
        print(f"Downloading {name} (one time only) ...")
        tmp = path + ".part"

        def progress(blocks, block_size, total):
            if total > 0:
                print(f"\r  {min(100, blocks * block_size * 100 // total)}%", end="", flush=True)

        urllib.request.urlretrieve(BASE_URL + name, tmp, progress)
        os.replace(tmp, path)
        print()
    return (os.path.join(model_dir, MODEL_FILE), os.path.join(model_dir, VOICES_FILE))


VOICE_FILE = "voice.json"


def load_voice_settings():
    """Your chosen Baymax voice. `python voice_lab.py --choose NAME` writes voice.json;
    command-line flags still override it."""
    settings = {"voice": DEFAULT_VOICE, "speed": DEFAULT_SPEED, "pitch": 1.0, "softness": 0.0}
    try:
        with open(VOICE_FILE, encoding="utf-8") as f:
            settings.update({k: v for k, v in json.load(f).items() if k in settings})
    except (FileNotFoundError, ValueError):
        pass
    return settings


def shape_voice(audio, pitch=1.0, softness=0.0):
    """Change the character of a voice with plain numpy (no extra packages).
    pitch     below 1.0 = deeper and 'bigger' (0.9 is a good start). Resampling also moves the
              voice's resonances down, like a larger body would. The audio gets longer by
              1/pitch; make_synth() cancels that so the speaking speed stays the same.
    softness  0 to 1: how much of the harsh high-frequency brightness to smooth away."""
    out = np.asarray(audio, dtype=np.float32)
    if abs(pitch - 1.0) > 1e-3 and len(out) > 1:
        n_out = max(1, int(round(len(out) / pitch)))
        out = np.interp(np.linspace(0, len(out) - 1, n_out), np.arange(len(out)), out).astype(np.float32)
    taps = 1 + 2 * int(round(min(max(softness, 0.0), 1.0) * 3))      # 1, 3, 5 or 7 taps
    if taps > 1:
        kernel = np.hanning(taps + 2)[1:-1]
        out = np.convolve(out, kernel / kernel.sum(), mode="same").astype(np.float32)
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > 0.99:
        out = out / peak * 0.99
    return out


def resolve_voice(kokoro, spec):
    """spec is one voice name ('am_michael') or a blend ('am_michael:0.6,bm_george:0.4').
    A blend mixes the two voices' style vectors into a new voice that nobody else has."""
    names = kokoro.get_voices()
    pairs = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        name, _, weight = part.partition(":")
        if name not in names:
            raise SystemExit(f"Unknown voice '{name}'. Run: python speak.py --list-voices")
        pairs.append((name, float(weight) if weight else 1.0))
    if not pairs:
        raise SystemExit("Empty voice name.")
    if len(pairs) == 1 and ":" not in spec:
        return pairs[0][0]
    total = sum(w for _, w in pairs)
    style = sum((w / total) * np.asarray(kokoro.get_voice_style(n)) for n, w in pairs)
    return np.asarray(style, dtype=np.float32)


def load_kokoro(model_dir=DEFAULT_MODEL_DIR):
    """Load Kokoro. With onnxruntime-gpu it runs on the GPU; set BAYMAX_VOICE_CPU=1 to force the CPU."""
    import os
    try:
        import onnxruntime
        onnxruntime.set_default_logger_severity(3)      # errors only: the GPU build prints harmless 'Memcpy nodes' warnings
        available = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in available and not os.environ.get("BAYMAX_VOICE_CPU"):
            preload = getattr(onnxruntime, "preload_dlls", None)
            if preload:
                preload()                     # finds libcufft, libcudnn ... inside the venv's nvidia/ packages
            # Ask for CUDA only. kokoro-onnx otherwise tries TensorRT first, which is not installed, prints a long
            # error, and retries.
            os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")
        elif os.environ.get("BAYMAX_VOICE_CPU"):
            os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
    except Exception:
        pass
    from kokoro_onnx import Kokoro
    model_path, voices_path = ensure_model_files(model_dir)
    kokoro = Kokoro(model_path, voices_path)
    try:                                      # what it REALLY runs on: a failed GPU load falls back silently
        print(f"Voice runs on: {kokoro.sess.get_providers()[0]}")
    except Exception:
        pass
    return kokoro


def make_synth(kokoro, voice=DEFAULT_VOICE, speed=DEFAULT_SPEED, pitch=1.0, softness=0.0):
    """Returns synth(text) -> (audio, sample_rate) for one voice setting."""
    voice_obj = resolve_voice(kokoro, voice)
    # Lowering the pitch stretches the audio afterwards. Asking Kokoro to speak faster by the
    # same factor cancels the stretch, so deeper does not mean slower.
    engine_speed = min(2.0, max(0.5, speed / pitch))

    def synth(text):
        audio, sample_rate = kokoro.create(text, voice=voice_obj, speed=engine_speed, lang="en-us")
        return shape_voice(audio, pitch, softness), sample_rate

    return synth


def load_tts(model_dir=DEFAULT_MODEL_DIR, voice=DEFAULT_VOICE, speed=DEFAULT_SPEED,
             pitch=1.0, softness=0.0):
    """Returns (synth, sample_rate, voice_names)."""
    kokoro = load_kokoro(model_dir)
    return make_synth(kokoro, voice, speed, pitch, softness), 24000, kokoro.get_voices()


def play_audio(audio, sample_rate):
    import sounddevice as sd
    sd.play(audio, sample_rate)
    sd.wait()


def save_wav(path, audio, sample_rate):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


# ---------------------------------------------------------------
# 2. THE SPEECH QUEUE  (synthesize ahead while playing)
#    synth_fn and play_fn are passed in, so the same class works with a fake
#    synth/player in tests.
# ---------------------------------------------------------------
class SpeechQueue:
    def __init__(self, synth_fn, play_fn=play_audio):
        self.synth_fn, self.play_fn = synth_fn, play_fn
        self._text_q = queue.Queue()
        self._audio_q = queue.Queue(maxsize=3)      # synthesize at most 3 sentences ahead
        self._pending = 0
        self._cond = threading.Condition()
        self._t0 = None
        self._first_audio = None
        self.synth_ms = []                          # how long each sentence took to synthesize
        self._threads = [
            threading.Thread(target=self._synth_loop, daemon=True),
            threading.Thread(target=self._play_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    # ----- public -------------------------------------------------
    def start_timer(self):
        """Call when the user finished speaking/typing. Time-to-first-audio counts from here."""
        self._t0 = time.perf_counter()
        self._first_audio = None
        self.synth_ms = []

    def say(self, text):
        with self._cond:
            self._pending += 1
        self._text_q.put(text)

    def wait(self, timeout=60):
        """Block until everything queued has finished playing. Gives up after `timeout` seconds instead of
        freezing the program forever if a worker thread got stuck. Returns True when everything finished."""
        end = time.monotonic() + timeout
        with self._cond:
            while self._pending > 0:
                left = end - time.monotonic()
                if left <= 0:
                    print("[speech: the voice did not finish in time, carrying on]", file=sys.stderr)
                    self._pending = 0
                    return False
                self._cond.wait(left)
        return True

    def ttfa_ms(self):
        """Milliseconds from start_timer() until the first sound started. None if no sound yet."""
        if self._t0 is None or self._first_audio is None:
            return None
        return (self._first_audio - self._t0) * 1000

    def close(self):
        self._text_q.put(None)
        for t in self._threads:
            t.join(timeout=5)

    # ----- worker threads -----------------------------------------
    def _done(self):
        with self._cond:
            self._pending = max(0, self._pending - 1)
            self._cond.notify_all()

    def _synth_loop(self):
        while True:
            text = self._text_q.get()
            if text is None:
                self._audio_q.put(None)
                return
            try:
                t = time.perf_counter()
                audio = self.synth_fn(text)
                self.synth_ms.append((time.perf_counter() - t) * 1000)
                self._audio_q.put(audio)
            except Exception as e:                  # one bad sentence must not stop the rest
                print(f"[speech error: {e}]", file=sys.stderr)
                self._done()

    def _play_loop(self):
        while True:
            item = self._audio_q.get()
            if item is None:
                return
            if self._first_audio is None:
                self._first_audio = time.perf_counter()
            try:
                self.play_fn(*item)
            except Exception as e:
                print(f"[playback error: {e}]", file=sys.stderr)
            self._done()


# ---------------------------------------------------------------
# 3. COMMAND LINE
# ---------------------------------------------------------------
def main():
    settings = load_voice_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="what to say")
    ap.add_argument("--voice", default=settings["voice"],
                    help="a voice name, or a blend like am_michael:0.6,bm_george:0.4")
    ap.add_argument("--speed", type=float, default=settings["speed"])
    ap.add_argument("--pitch", type=float, default=settings["pitch"], help="below 1.0 = deeper (try 0.9)")
    ap.add_argument("--softness", type=float, default=settings["softness"], help="0 to 1 (try 0.3)")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--save", metavar="FILE.wav", help="write a wav file instead of playing")
    args = ap.parse_args()

    synth, sample_rate, names = load_tts(args.model_dir, args.voice, args.speed, args.pitch, args.softness)
    if args.list_voices:
        print(", ".join(n for n in names if n[:2] in ("af", "am", "bf", "bm")))
        return

    text = speakable(" ".join(args.text) or "Hello, I am Baymax, your personal healthcare companion.")
    t = time.perf_counter()
    audio, sample_rate = synth(text)
    took = time.perf_counter() - t
    seconds = len(audio) / sample_rate
    print(f"Synthesis took {took * 1000:.0f} ms for {seconds:.1f} s of speech (real-time factor {took / seconds:.2f}).")
    print("A factor under 1.0 means it is faster than real time. Phase 5 wants the first sentence in about 150 ms.")

    if args.save:
        save_wav(args.save, audio, sample_rate)
        print(f"Saved {args.save}")
    else:
        play_audio(audio, sample_rate)


if __name__ == "__main__":
    main()
