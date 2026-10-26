"""
baymax_live.py - Baymax wakes up when you say "ouch", "it hurts", "oh no" ...

    mic -> VAD -> Whisper -> text
             |
             v
       Session (this file)
         SLEEPING: ignore everything, unless
                   - the safety gate fires (chest pain, self-harm ...)  -> fixed reply, wake up
                   - a wake phrase is heard (activation.py)             -> "On a scale of 1 to 10 ..."
         ACTIVE:   gate -> model -> guard -> voice, with a little chat history
                   "thank you" / "I am satisfied" / silence for a while -> back to sleep

Run:
    python baymax_live.py                 # microphone
    python baymax_live.py --typed         # type instead of speaking (test the logic, no mic)
    python baymax_live.py --typed --no-audio
    python baymax_live.py --idle 45       # seconds of silence before Baymax goes back to sleep

The Session class has no audio, model or microphone code inside it (they are passed in),
so test_activation.py can test it with fakes.
"""

import argparse
import queue
import sys
import time

from activation import FRIEND_SYSTEM, SLEEP_LINE, TriggerDetector, is_filler, is_sleep_phrase, opening_line
from safety_gate import check_message

SLEEPING, ACTIVE = "sleeping", "active"
CARE, FRIEND = "care", "friend"      # care = gentle health helper, friend = casual chat
CARE_KINDS = {"pain", "distress", "help"}

IDLE_SECONDS = 30          # active this long without hearing you -> back to sleep
HISTORY_MESSAGES = 6       # remember the last 3 exchanges, no more (the model is small)


class Session:
    """State machine. handle(text) returns a short label saying what happened:
    'ignored', 'asleep', 'woke', 'gate', 'sleep', 'reply'."""

    def __init__(self, reply, say_fixed, detector=None, idle_seconds=IDLE_SECONDS,
                 clock=time.monotonic, log=print):
        self.reply = reply                   # reply(text, history, mode) -> list of sentences spoken
        self.say_fixed = say_fixed           # say_fixed(text): speak a pre-written line
        self.detector = detector or TriggerDetector()
        self.idle_seconds = idle_seconds
        self.clock = clock
        self.log = log
        self.state = SLEEPING
        self.mode = CARE
        self.history = []
        self.last_heard = 0.0

    # -- helpers ------------------------------------------------
    def _remember(self, user, assistant):
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        self.history = self.history[-HISTORY_MESSAGES:]

    def _wake(self):
        self.state = ACTIVE
        self.last_heard = self.clock()

    def _sleep(self):
        self.state = SLEEPING
        self.history = []

    def expire_if_idle(self):
        """Called before every utterance. Silent drift back to sleep (no speech, it is rude to interrupt)."""
        if self.state == ACTIVE and self.clock() - self.last_heard > self.idle_seconds:
            self.log("   [no voice for a while: Baymax went back to sleep]")
            self._sleep()

    # -- the one entry point ------------------------------------
    def handle(self, text):
        self.expire_if_idle()
        if is_filler(text):
            return "ignored"

        gate = check_message(text)
        if gate.triggered:                                   # safety ALWAYS wins, awake or not
            self.log(f"   [gate: {gate.category}. The model was NOT used.]")
            self._wake()
            self.say_fixed(gate.message)
            self._remember(text, gate.message)
            return "gate"

        if self.state == SLEEPING:
            trig = self.detector.check(text)
            if trig is None:
                return "asleep"
            self.log(f"   [wake: {trig.kind} phrase '{trig.phrase}']")
            self.mode = CARE if trig.kind in CARE_KINDS else FRIEND
            self._wake()
            line = opening_line(trig.kind)
            self.say_fixed(line)
            self._remember(text, line)
            return "woke"

        # active
        self.last_heard = self.clock()
        if is_sleep_phrase(text):
            self.say_fixed(SLEEP_LINE)
            self._sleep()
            return "sleep"
        trig = self.detector.check(text)
        if trig and trig.kind in CARE_KINDS:                 # pain talk in the middle of a chat: be careful again
            self.mode = CARE
        sentences = self.reply(text, list(self.history), self.mode)
        self._remember(text, " ".join(sentences) if sentences else "")
        self.last_heard = self.clock()                       # the clock starts after Baymax finishes talking
        return "reply"


# ---------------------------------------------------------------
# Wiring: real model + real voice
# ---------------------------------------------------------------
def build_speaker(args):
    """Returns (say, wait, close, start_timer)."""
    if args.no_audio:
        return (lambda t: None), (lambda: None), (lambda: None), (lambda: None)
    from speak import SpeechQueue, load_tts
    print("Loading the voice ...")
    synth, _, _ = load_tts(args.model_dir, args.voice, args.speed, args.pitch, args.softness)
    q = SpeechQueue(synth)
    return q.say, q.wait, q.close, q.start_timer


def make_session(args):
    from baymax_speak import model_tokens, speak_stream
    say, wait, close, start_timer = build_speaker(args)
    show = lambda s: print("Baymax:", s)

    def say_fixed(text):
        start_timer()
        speak_stream([text + " "], say, guard=False, show=show)
        wait()

    def reply(text, history, mode=CARE):
        start_timer()
        marks, t0 = {}, time.perf_counter()
        friend = mode == FRIEND
        tokens = model_tokens(text, marks, history, system=FRIEND_SYSTEM if friend else None,
                              temperature=0.7 if friend else 0.4)
        stats = speak_stream(tokens, say, show=show)
        for reason, sentence in stats["removed"]:
            print(f"   [guard removed ({reason}): {sentence}]")
        wait()
        if marks.get("first_token"):
            print(f"   [model first token {(marks['first_token'] - t0) * 1000:.0f} ms]")
        return stats["sentences"]

    return Session(reply, say_fixed, idle_seconds=args.idle), close


# ---------------------------------------------------------------
# Typed mode (no microphone) and microphone mode
# ---------------------------------------------------------------
def run_typed(session):
    print("Typed mode. Baymax is asleep: type 'ouch' or 'it hurts' to wake it. 'quit' to stop.\n")
    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            return
        if text.lower() in ("quit", "exit", "q"):
            return
        if text:
            result = session.handle(text)
            print(f"   [{result} | state: {session.state}]\n")


def run_mic(session, args):
    from listen import (CHUNK, MIN_SPEECH_SEC, SAMPLE_RATE, UtteranceBuffer, ensure_cuda_libs_on_path,
                        load_stt, load_vad, transcribe)
    import sounddevice as sd
    import torch

    print("Loading Silero VAD ...")
    vad = load_vad(args.threshold)
    stt, stt_device = load_stt(args.stt_device)
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    buf = UtteranceBuffer()

    def forget_audio():
        """While Baymax talks, the mic hears Baymax. After each reply, throw away everything
        recorded meanwhile so it never answers itself."""
        while True:
            try:
                audio_q.get_nowait()
            except queue.Empty:
                break
        buf.reset()
        vad.reset_states()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK, device=args.mic, callback=callback):
        print("\nBaymax is asleep. Say 'ouch', 'it hurts' or 'oh no'. Ctrl+C to stop.\n")
        while True:
            chunk = audio_q.get()
            audio = buf.feed(chunk, vad(torch.from_numpy(chunk)))
            if audio is None:
                continue
            vad.reset_states()
            if len(audio) / SAMPLE_RATE < MIN_SPEECH_SEC:
                continue
            text = transcribe(stt, audio)
            if not text:
                continue
            print(f'You said: "{text}"')
            result = session.handle(text)
            if result == "asleep":
                print("   [asleep: ignored]")
            if result != "ignored" and result != "asleep":
                forget_audio()                              # Baymax just spoke: drop its own echo
            print()


def main():
    from speak import DEFAULT_MODEL_DIR, load_voice_settings
    settings = load_voice_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--typed", action="store_true", help="type instead of using the microphone")
    ap.add_argument("--no-audio", action="store_true", help="do not load or play the voice")
    ap.add_argument("--idle", type=float, default=IDLE_SECONDS, help="seconds before it goes back to sleep")
    ap.add_argument("--stt-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--mic", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--voice", default=settings["voice"])
    ap.add_argument("--speed", type=float, default=settings["speed"])
    ap.add_argument("--pitch", type=float, default=settings["pitch"])
    ap.add_argument("--softness", type=float, default=settings["softness"])
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    args = ap.parse_args()

    if not args.typed and args.stt_device == "cuda":
        from listen import ensure_cuda_libs_on_path
        ensure_cuda_libs_on_path()

    session, close = make_session(args)
    try:
        run_typed(session) if args.typed else run_mic(session, args)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        close()


if __name__ == "__main__":
    main()
