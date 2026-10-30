"""
benchmark.py - Phase 5: how fast is the WHOLE Baymax pipeline, stage by stage, against the roadmap goals?

It needs no microphone and no speaker. For each test sentence it:
    1. makes "speech" with the Kokoro voice (a different voice from Baymax's own) and resamples it to 16 kHz,
    2. runs Faster-Whisper on it                       (speech-to-text)
    3. hands the text to the real Session              (safety gate -> model -> output guard -> sentence buffer)
    4. synthesizes the reply with the real voice and stops at the first sound (nothing is played aloud)

Run:
    python benchmark.py                       # Whisper on the GPU, 3 runs per sentence
    python benchmark.py --stt-device cpu --runs 5
    python benchmark.py --online-friend       # also time the Groq/OpenRouter path for casual chat

Honest limits, so you read the numbers correctly:
  * The "speech" is synthetic and clean, so speech-to-text is a little faster and more accurate than with your real
    voice and room noise. Treat the STT row as a best case; run  python listen.py  to see real timings.
  * The VAD silence wait (600 ms) is not included; latency.py explains how to add it.
  * Playback device buffering (tens of ms) is not included.
  * The first round is a warm-up and is thrown away (model loading, GPU kernel setup).
"""

import argparse
import contextlib
import io
import time
from types import SimpleNamespace

import numpy as np

from latency import Summary, Trace

# (kind, what the person says). "gate" lines are answered by the safety rules, never by the model.
UTTERANCES = [
    ("friend", "hello there how are you today"),
    ("friend", "i finally finished my big project at work"),
    ("care", "i have a sore throat, is there any home remedy"),
    ("care", "what is the difference between a cold and the flu"),
    ("care", "any tips for hiccups"),
    ("gate", "my chest feels very tight and i am sweating a lot"),
]
SPEECH_VOICE = "af_heart"      # plays the "user"; Baymax answers in the voice from voice.json


def to_16k(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if sr == 16000:
        return audio
    try:
        from scipy.signal import resample_poly
        return resample_poly(audio, 16000, sr).astype(np.float32)
    except ImportError:
        n = int(len(audio) * 16000 / sr)
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def make_awake(session):
    """Put the Session in the state it has in the middle of a conversation: awake, empty history."""
    from baymax_live import ACTIVE, CARE
    session.state, session.mode, session.care_hold = ACTIVE, CARE, 0
    session.intake, session.history, session.rating_tries = None, [], 0
    session.last_heard = session.clock()


def main():
    from speak import DEFAULT_MODEL_DIR, load_voice_settings
    settings = load_voice_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="timed runs per sentence (after one warm-up round)")
    ap.add_argument("--stt-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--online-friend", action="store_true")
    ap.add_argument("--fast-first", action="store_true", help="speak the first clause of model replies early")
    ap.add_argument("--no-cache", action="store_true", help="ignore pre-rendered audio (to see the difference)")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--voice-backend", default="xtts", choices=["kokoro", "voicekit", "xtts"],
                    help="which voice answers (the app defaults to xtts)")
    ap.add_argument("--clone-voice", default="baymax", help="voice profile for xtts / voicekit")
    a = ap.parse_args()

    if a.stt_device == "cuda":
        from listen import ensure_cuda_libs_on_path
        ensure_cuda_libs_on_path()                         # may restart this script once with CUDA libraries on the path

    from baymax_live import make_session
    from listen import load_stt, transcribe
    from speak import load_tts

    args = SimpleNamespace(typed=True, no_audio=False, silent=True, online_friend=a.online_friend, no_memory=True,
                           memory=":memory:", idle=10_000, stt_device=a.stt_device, model_dir=a.model_dir,
                           voice=settings["voice"], speed=settings["speed"], pitch=settings["pitch"],
                           softness=settings["softness"], no_cache=a.no_cache, fast_first=a.fast_first, audio_cache="audio_cache",
                           voice_backend=a.voice_backend, clone_voice=a.clone_voice)
    print("Loading speech-to-text ...")
    stt, stt_device = load_stt(a.stt_device)
    print("Loading the pipeline (voice + model) ...")
    session, close = make_session(args)
    print(f"Making the test 'speech' with voice {SPEECH_VOICE} ...")
    speech_synth, _, _ = load_tts(a.model_dir, SPEECH_VOICE, 1.0, 1.0, 0.0)
    clips = {text: to_16k(*speech_synth(text)) for _, text in UTTERANCES}

    overall, by_kind = Summary(), {}
    try:
        for round_no in range(a.runs + 1):
            warm = round_no == 0
            print(f"\n{'Warm-up round (not counted)' if warm else f'Run {round_no} of {a.runs}'}")
            for kind, text in UTTERANCES:
                make_awake(session)
                session.marks.clear()
                t_end = time.perf_counter()
                heard = transcribe(stt, clips[text])
                t_stt = time.perf_counter()
                session.last_heard = session.clock()
                with contextlib.redirect_stdout(io.StringIO()):          # keep the table readable
                    result = session.handle(heard)
                trace = Trace(end_of_speech=t_end, stt_done=t_stt, **session.marks)
                total = trace.total_ms()
                print(f"  {kind:6} heard: {heard!r:62} -> {result:6} "
                      f"{'%.0f ms' % total if total is not None else 'no sound'}")
                if not warm:
                    overall.add(trace)
                    by_kind.setdefault(kind, Summary()).add(trace)
    finally:
        close()

    print(f"\n=== ALL TURNS (speech-to-text on {stt_device}) ===\n{overall.table()}\n{overall.verdict()}")
    for kind, s in by_kind.items():
        print(f"\n--- {kind} ---\n{s.table()}")
    from audio_cache import AudioCache
    t = AudioCache.totals
    if t["hits"] + t["misses"]:
        print(f"\nPre-rendered audio used for {t['hits']} of {t['hits'] + t['misses']} spoken chunks "
              f"({100 * t['hits'] // (t['hits'] + t['misses'])}%). Misses were synthesized live.")
    print("\nReminder: synthetic speech makes the speech-to-text row a best case. The 600 ms silence wait is not included.")


if __name__ == "__main__":
    main()
