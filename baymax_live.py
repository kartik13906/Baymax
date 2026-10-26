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

from activation import (ASK_RATING, ASK_RATING_AGAIN, FRIEND_SYSTEM, HIGH_PAIN_AT, HIGH_PAIN_LINE, SLEEP_LINE,
                        TriggerDetector, has_cause, is_filler, is_health_topic, is_scale_question, is_sleep_phrase,
                        opening_line, parse_pain_rating)
from safety_gate import check_message

SLEEPING, ACTIVE = "sleeping", "active"
CARE, FRIEND = "care", "friend"      # care = gentle health helper, friend = casual chat
CARE_KINDS = {"pain", "distress", "help"}

IDLE_SECONDS = 30          # active this long without hearing you -> back to sleep
HISTORY_MESSAGES = 6       # remember the last 3 exchanges, no more (the model is small)
CARE_HOLD_TURNS = 3        # after health talk, the next few messages ("yes, since Monday") also stay local


class Session:
    """State machine. handle(text) returns a short label saying what happened:
    'ignored', 'asleep', 'woke', 'gate', 'sleep', 'reply'."""

    def __init__(self, reply, say_fixed, detector=None, idle_seconds=IDLE_SECONDS,
                 clock=time.monotonic, log=print):
        self.reply = reply                   # reply(text, history, mode, skip_scale) -> list of sentences spoken
        self.say_fixed = say_fixed           # say_fixed(text): speak a pre-written line
        self.detector = detector or TriggerDetector()
        self.idle_seconds = idle_seconds
        self.clock = clock
        self.log = log
        self.state = SLEEPING
        self.mode = CARE
        self.care_hold = 0                   # turns left in which short follow-ups still count as health talk
        self.intake = None                   # pain check-in: None, "cause" (what/where?) or "rating" (how bad?)
        self.cause = ""                      # what the person said hurts / happened
        self.rating_tries = 0
        self.history = []
        self.last_heard = 0.0

    # -- helpers ------------------------------------------------
    def _remember(self, user, assistant):
        # every turn is tagged with its mode, so only FRIEND turns can ever be sent to an online provider
        self.history += [{"role": "user", "content": user, "mode": self.mode},
                         {"role": "assistant", "content": assistant, "mode": self.mode}]
        self.history = self.history[-HISTORY_MESSAGES:]

    def _wake(self):
        self.state = ACTIVE
        self.last_heard = self.clock()

    def _sleep(self):
        self.state = SLEEPING
        self.history = []
        self.care_hold = 0
        self.intake, self.cause, self.rating_tries = None, "", 0

    def _route(self, text, trig=None):
        """Pick the mode for THIS message. Health words -> CARE (local model only), and CARE stays on
        for a few follow-up messages. Everything else -> FRIEND."""
        if (trig and trig.kind in CARE_KINDS) or is_health_topic(text):
            self.mode, self.care_hold = CARE, CARE_HOLD_TURNS
        elif self.care_hold > 0:
            self.care_hold -= 1                              # mode stays CARE
        else:
            self.mode = FRIEND

    def expire_if_idle(self):
        """Called before every utterance. Silent drift back to sleep (no speech, it is rude to interrupt)."""
        if self.state == ACTIVE and self.clock() - self.last_heard > self.idle_seconds:
            self.log("   [no voice for a while: Baymax went back to sleep]")
            self._sleep()

    def _check_in(self, text):
        """Pain check-in. Step 1 'cause': what hurts / what happened. Step 2 'rating': how bad, 1 to 10."""
        self.mode, self.care_hold = CARE, CARE_HOLD_TURNS
        if self.intake == "cause":
            self.cause = text
            rating = parse_pain_rating(text, strict=True)    # they may have said "my arm, 7 out of 10" already
            if rating is None:
                self.intake = "rating"
                self.say_fixed(ASK_RATING)
                self._remember(text, ASK_RATING)
                return "intake"
            return self._finish_check_in(rating)
        rating = parse_pain_rating(text)
        if rating is None and self.rating_tries == 0:
            self.rating_tries = 1                            # one polite retry, then carry on without a number
            self.say_fixed(ASK_RATING_AGAIN)
            self._remember(text, ASK_RATING_AGAIN)
            return "intake"
        return self._finish_check_in(rating)

    def _finish_check_in(self, rating):
        """We know what happened and how bad it is. Tell the model, so it does not ask again."""
        cause = self.cause.strip().rstrip(".")
        self.intake, self.rating_tries = None, 0
        if rating is None:
            prompt = f"I am in pain. {cause}."
        else:
            prompt = f"I am in pain. {cause}. My pain is {rating} out of 10."
        if rating is not None and rating >= HIGH_PAIN_AT:
            self.say_fixed(HIGH_PAIN_LINE)
        sentences = self.reply(prompt, list(self.history), CARE, True)
        self._remember(prompt, " ".join(sentences) if sentences else "")
        self.last_heard = self.clock()
        return "reply"

    # -- the one entry point ------------------------------------
    def handle(self, text):
        self.expire_if_idle()
        if is_filler(text):
            return "ignored"

        gate = check_message(text)
        if gate.triggered:                                   # safety ALWAYS wins, awake or not
            self.log(f"   [gate: {gate.category}. The model was NOT used.]")
            self._wake()
            self.mode, self.care_hold = CARE, CARE_HOLD_TURNS
            self.say_fixed(gate.message)
            self._remember(text, gate.message)
            return "gate"

        if self.state == SLEEPING:
            trig = self.detector.check(text)
            if trig is None:
                return "asleep"
            self.log(f"   [wake: {trig.kind} phrase '{trig.phrase}']")
            self.care_hold = 0
            self._route(text, trig)
            self._wake()
            line = opening_line(trig.kind)
            if trig.kind == "pain":
                if has_cause(text, trig.phrase):               # "ouch my arm hurts": we know where already
                    self.cause = text
                    line = ASK_RATING
                    self.intake = "rating"
                else:
                    self.intake = "cause"
            self.say_fixed(line)
            self._remember(text, line)
            return "woke"

        # active
        self.last_heard = self.clock()
        if is_sleep_phrase(text):
            self.say_fixed(SLEEP_LINE)
            self._sleep()
            return "sleep"
        if self.intake:
            return self._check_in(text)
        self._route(text, self.detector.check(text))         # pain or health talk mid-chat: back to the local model
        sentences = self.reply(text, list(self.history), self.mode)
        self._remember(text, " ".join(sentences) if sentences else "")
        self.last_heard = self.clock()                       # the clock starts after Baymax finishes talking
        return "reply"


# ---------------------------------------------------------------
# What may leave this computer
# ---------------------------------------------------------------
def plain(history):
    """Messages in the plain {"role", "content"} form that model APIs accept (drops our "mode" tag)."""
    return [{"role": h["role"], "content": h["content"]} for h in history]


def api_messages(text, history):
    """The ONLY function that builds what is sent to an online provider: the friend prompt, the earlier
    FRIEND-mode turns, and the new message. Turns tagged CARE (health talk, safety-gate replies) are
    filtered out here, so they can never be sent online, even in a later casual message."""
    friend_turns = [h for h in history if h.get("mode") == FRIEND]
    return [{"role": "system", "content": FRIEND_SYSTEM}] + plain(friend_turns) + [{"role": "user", "content": text}]


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

    chain = None
    if getattr(args, "online_friend", False):
        from llm_backends import build_chain, load_env_file
        load_env_file()
        chain = build_chain()
        print("Online friend chat is ON: casual messages go to an online provider.")
        print("Health talk, safety replies and anything about pain stay on this computer.")
        print("Providers:", chain.describe(), "\n")

    def friend_tokens(text, history, marks, source):
        """Casual chat: try Groq, then OpenRouter, then fall back to the local friend model."""
        from llm_backends import NoBackendAvailable
        try:
            for piece in chain.stream(api_messages(text, history)):
                marks.setdefault("first_token", time.perf_counter())
                source["name"] = chain.last_used
                yield piece
        except NoBackendAvailable:
            print("   [no online provider available: using the local model]")
            source["name"] = "local"
            yield from model_tokens(text, marks, plain(history), system=FRIEND_SYSTEM, temperature=0.7)

    def reply(text, history, mode=CARE, skip_scale=False):
        start_timer()
        marks, t0, source = {}, time.perf_counter(), {"name": "local"}
        friend = mode == FRIEND
        if friend and chain:
            tokens = friend_tokens(text, history, marks, source)
        else:
            tokens = model_tokens(text, marks, plain(history), system=FRIEND_SYSTEM if friend else None,
                                  temperature=0.7 if friend else 0.4)
        stats = speak_stream(tokens, say, show=show, drop=is_scale_question if skip_scale else None)
        for reason, sentence in stats["removed"]:
            print(f"   [guard removed ({reason}): {sentence}]")
        wait()
        if marks.get("first_token"):
            print(f"   [{mode} mode | answered by: {source['name']} | first token {(marks['first_token'] - t0) * 1000:.0f} ms]")
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
    ap.add_argument("--online-friend", action="store_true",
                    help="use Groq, then OpenRouter, for casual chat (needs API keys; see llm_backends.py)")
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
