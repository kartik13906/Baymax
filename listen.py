"""
listen.py - Baymax Phase 3, step 1: microphone -> VAD -> speech-to-text.

    mic (16 kHz) -> Silero VAD (finds where you start/stop talking)
                 -> Faster-Whisper distil-medium.en (turns that speech into text)
                 -> safety gate check (just displayed for now)

Run:
    python listen.py                      # Whisper on the GPU
    python listen.py --stt-device cpu     # if the GPU path fails
    python listen.py --list-devices       # show microphones
    python listen.py --mic 3              # use microphone number 3

Speak a sentence, then pause. After about 600 ms of silence the speech is sent to
Whisper and the text appears, with timings you will need for Phase 5 benchmarking.
"""

import argparse
import queue
import sys
import time
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
CHUNK = 512            # Silero VAD needs exactly 512 samples per call at 16 kHz (= 32 ms)
SILENCE_MS = 600       # how long a pause counts as "finished talking" (your roadmap's value)
PRE_ROLL_CHUNKS = 10   # keep 10 x 32 ms = 320 ms from BEFORE speech was detected, so
                       # the first syllable is not cut off
MIN_SPEECH_SEC = 0.3   # ignore shorter blips (clicks, coughs)
MAX_SPEECH_SEC = 30    # safety cap: force a cut if someone talks non-stop


# ---------------------------------------------------------------
# 1. UTTERANCE BUFFER
#    The VAD only says "speech started" / "speech ended". This small class
#    does the bookkeeping: keep a rolling pre-roll, collect chunks while
#    speaking, and hand back the whole utterance when it ends.
#    It has no audio or model code in it, so it is easy to test.
# ---------------------------------------------------------------
class UtteranceBuffer:
    def __init__(self, pre_roll_chunks=PRE_ROLL_CHUNKS, max_sec=MAX_SPEECH_SEC):
        self.pre_roll = deque(maxlen=pre_roll_chunks)
        self.chunks = []
        self.in_speech = False
        self.max_chunks = int(max_sec * SAMPLE_RATE / CHUNK)

    def feed(self, chunk, event):
        """chunk: float32 array of 512 samples.
        event: result of Silero's VADIterator: None, {'start': ...} or {'end': ...}.
        Returns the finished utterance (numpy array) or None."""
        if not self.in_speech:
            self.pre_roll.append(chunk)
            if event and "start" in event:
                self.in_speech = True
                self.chunks = list(self.pre_roll)      # begin with the pre-roll audio
            return None

        self.chunks.append(chunk)
        if (event and "end" in event) or len(self.chunks) >= self.max_chunks:
            audio = np.concatenate(self.chunks)
            self.reset()
            return audio
        return None

    def reset(self):
        self.pre_roll.clear()
        self.chunks = []
        self.in_speech = False


# ---------------------------------------------------------------
# 2. LOADING THE MODELS
#    Heavy imports happen inside the functions so the file can be
#    imported (and the buffer tested) without torch or a GPU.
# ---------------------------------------------------------------
def load_vad(threshold):
    from silero_vad import VADIterator, load_silero_vad
    model = load_silero_vad()
    return VADIterator(
        model,
        threshold=threshold,                 # higher = stricter about what counts as speech
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=SILENCE_MS,  # the 600 ms end-of-speech boundary
        speech_pad_ms=100,
    )


def ensure_cuda_libs_on_path():
    """ctranslate2 (the engine under Faster-Whisper) looks for the CUDA 12 libraries
    libcublas.so.12 and libcudnn*.so.9. Arch's system CUDA is newer and does not
    have them, but `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` puts copies
    inside the venv. LD_LIBRARY_PATH is only read when a program STARTS, so if we
    find those folders and the variable is not set yet, we set it and restart
    this script once."""
    import importlib.util
    import os
    if os.environ.get("BAYMAX_LD_SET"):
        return                                   # already restarted once
    dirs = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            for base in spec.submodule_search_locations:
                lib = os.path.join(base, "lib")
                if os.path.isdir(lib):
                    dirs.append(lib)
    if not dirs:
        return                                   # packages not installed: nothing to do
    os.environ["BAYMAX_LD_SET"] = "1"
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([existing] if existing else []))
    os.execv(sys.executable, [sys.executable] + sys.argv)


def load_stt(device):
    from faster_whisper import WhisperModel

    def build(dev):
        # int8_float16 keeps the model small on the GPU (your 6 GB also has to hold the
        # Baymax LLM and the voice model later). On CPU, plain int8 is the fast choice.
        compute_type = "int8_float16" if dev == "cuda" else "int8"
        print(f"Loading Whisper distil-medium.en on {dev} ({compute_type}) ...")
        model = WhisperModel("distil-medium.en", device=dev, compute_type=compute_type)
        # Warm-up: the first call is slow (kernel setup), and a missing CUDA library
        # shows up HERE, early, instead of in the middle of your first sentence.
        segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")
        list(segments)
        return model

    try:
        model = build(device)
    except Exception as e:
        if device == "cpu":
            raise
        print(f"\n!! Whisper could not run on the GPU: {e}")
        print("!! Falling back to the CPU so you can keep going. To fix the GPU, run:")
        print("!!     pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n")
        device = "cpu"
        model = build(device)
    print(f"Whisper ready on {device}.")
    return model, device


def load_gate():
    try:
        from safety_gate import check_message
        return check_message
    except ImportError:
        return None


def transcribe(model, audio):
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=1,                        # greedy decoding: fastest
        without_timestamps=True,
        condition_on_previous_text=False,   # each utterance stands alone
    )
    return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------
# 3. THE LISTENING LOOP
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--mic", type=int, default=None, help="input device number")
    ap.add_argument("--threshold", type=float, default=0.5, help="VAD sensitivity, 0.3-0.7")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.stt_device == "cuda" and not args.list_devices:
        ensure_cuda_libs_on_path()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return

    import torch

    print("Loading Silero VAD ...")
    vad = load_vad(args.threshold)
    stt, stt_device = load_stt(args.stt_device)
    check_message = load_gate()

    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        # Runs in the audio thread: do the minimum, hand the chunk to the main loop.
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    buf = UtteranceBuffer()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK, device=args.mic, callback=callback):
        print("\nListening. Speak a sentence, then pause. Ctrl+C to stop.\n")
        try:
            while True:
                chunk = audio_q.get()
                event = vad(torch.from_numpy(chunk))
                was_idle = not buf.in_speech
                audio = buf.feed(chunk, event)

                if was_idle and buf.in_speech:
                    print("  ... hearing speech", flush=True)

                if audio is None:
                    continue

                vad.reset_states()                 # resync the VAD after every utterance
                seconds = len(audio) / SAMPLE_RATE
                if seconds < MIN_SPEECH_SEC:
                    continue

                t0 = time.perf_counter()
                text = transcribe(stt, audio)
                stt_ms = (time.perf_counter() - t0) * 1000
                if not text:
                    continue

                print(f'You said: "{text}"')
                print(f"  speech {seconds:.1f} s | speech-to-text on {stt_device} took {stt_ms:.0f} ms "
                      f"(+ {SILENCE_MS} ms waiting for you to finish)")
                if check_message:
                    g = check_message(text)
                    print(f"  [gate: {g.category} -> fixed reply]" if g.triggered else "  [gate: clear]")
                print()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()