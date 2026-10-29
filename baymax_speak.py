"""
baymax_speak.py - Baymax answers OUT LOUD (typed input for now; the microphone joins in Phase 5).

    you type -> safety gate -----(emergency/crisis/diagnosis)-----> fixed reply spoken
                    |
                    +--(clear)--> Baymax model streams tokens
                                   -> sentence buffer (whole sentences)
                                   -> output guard checks each sentence (numbers, medicines, promises)
                                   -> voice speaks it

Run:     python baymax_speak.py
         python baymax_speak.py --voice af_heart --no-audio     # print only, no sound
Quit:    type "quit" or press Ctrl+C

It prints three timings after every answer, the numbers you need for Phase 5:
    model first token  /  first sentence ready  /  first sound
"""

import argparse
import time

from output_guard import DETAIL_NOTE, MAX_SENTENCES, SAFE_FALLBACK, check_sentence, strip_formatting
from safety_gate import check_message
from sentence_buffer import SentenceBuffer
from speak import DEFAULT_MODEL_DIR, SpeechQueue, load_voice_settings, speakable

MODEL = "baymax"
SPLIT_ON_COMMAS = False        # see the note in sentence_buffer.py before changing this
DETAIL_REASONS = {"number", "number word", "frequency", "medicine or test"}


def speak_stream(tokens, say, show=print, guard=True, drop=None, fast_first=False):
    """Read streamed text pieces, speak complete sentences.
    tokens: any iterable of strings.   say(text): speaks a sentence.   show(text): prints it.
    guard=False is ONLY for the fixed gate replies, which contain a phone number on purpose.
    drop(sentence) -> True silently skips a sentence (used to skip a repeated pain-scale question).
    Returns a dict describing what happened."""
    # fast_first only for guarded model replies: fixed lines are pre-rendered per sentence and must not be re-split
    buf = SentenceBuffer(split_on_commas=SPLIT_ON_COMMAS, first_clause_only=bool(fast_first and guard))
    stats = {"spoken": 0, "clauses": 0, "removed": [], "first_sentence_at": None, "sentences": []}

    def emit(sentence):
        """Show a sentence, speak it, and remember it (the live loop keeps it as chat history)."""
        if stats["first_sentence_at"] is None:
            stats["first_sentence_at"] = time.perf_counter()
        show(sentence)
        say(speakable(sentence))
        stats["sentences"].append(sentence)

    def handle(sentence):
        """Returns False when we should stop reading the stream."""
        sentence = strip_formatting(sentence)
        if not sentence:
            return True
        if drop and drop(sentence):
            return True
        if guard:
            reason = check_sentence(sentence)
            if reason:
                stats["removed"].append((reason, sentence))
                return True
        emit(sentence)
        stats["spoken"] += 1
        if sentence.endswith(","):
            stats["clauses"] += 1                                   # an early first clause is not a whole sentence
        return not (guard and stats["spoken"] - stats["clauses"] >= MAX_SENTENCES)

    keep_going = True
    for piece in tokens:
        for sentence in buf.feed(piece):
            if not handle(sentence):
                keep_going = False
                break
        if not keep_going:
            break
    if keep_going:
        for sentence in buf.flush():
            if not handle(sentence):
                break

    if guard:
        if stats["spoken"] == 0:
            emit(SAFE_FALLBACK)
        elif any(r in DETAIL_REASONS for r, _ in stats["removed"]):
            emit(DETAIL_NOTE)
    return stats


def model_tokens(user_text, marks, history=None, system=None, temperature=0.4):
    """Stream the answer from the Ollama model, recording when the first token arrives.
    history: earlier messages [{"role": "user"|"assistant", "content": ...}] so a short reply
    like "seven" still makes sense.
    system: replaces the Modelfile's system prompt for this call (used for the casual "friend" mode)."""
    import ollama
    stream = ollama.chat(
        model=MODEL,
        messages=([{"role": "system", "content": system}] if system else [])
                 + list(history or []) + [{"role": "user", "content": user_text}],   # no system: the Modelfile's is used
        options={"temperature": temperature, "num_predict": 150},
        stream=True,
    )
    for chunk in stream:
        if "first_token" not in marks:
            marks["first_token"] = time.perf_counter()
        yield chunk["message"]["content"]


def main():
    settings = load_voice_settings()                  # your choice from voice.json, if you made one
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=settings["voice"])
    ap.add_argument("--speed", type=float, default=settings["speed"])
    ap.add_argument("--pitch", type=float, default=settings["pitch"])
    ap.add_argument("--softness", type=float, default=settings["softness"])
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--no-audio", action="store_true", help="print only; skip loading the voice")
    args = ap.parse_args()

    if args.no_audio:
        speaker = None
        say = lambda text: None
    else:
        from speak import load_tts
        print("Loading the voice ...")
        synth, _, _ = load_tts(args.model_dir, args.voice, args.speed, args.pitch, args.softness)
        speaker = SpeechQueue(synth)
        say = speaker.say

    print("Baymax is ready. Type a message (or 'quit').\n")
    try:
        while True:
            try:
                text = input("You: ").strip()
            except EOFError:
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue

            t0 = time.perf_counter()
            marks = {}
            if speaker:
                speaker.start_timer()

            gate = check_message(text)
            if gate.triggered:
                print(f"   [gate: {gate.category}. The model was NOT used.]")
                stats = speak_stream([gate.message + " "], say, guard=False, show=lambda s: print("Baymax:", s))
            else:
                stats = speak_stream(model_tokens(text, marks), say, show=lambda s: print("Baymax:", s))
                for reason, sentence in stats["removed"]:
                    print(f"   [guard removed ({reason}): {sentence}]")

            if speaker:
                speaker.wait()

            def ms(t):
                return f"{(t - t0) * 1000:.0f} ms" if t else "n/a"
            first_sound = f"{speaker.ttfa_ms():.0f} ms" if speaker and speaker.ttfa_ms() else "n/a"
            print(f"   [timing: model first token {ms(marks.get('first_token'))} | "
                  f"first sentence ready {ms(stats['first_sentence_at'])} | first sound {first_sound}]\n")
    except KeyboardInterrupt:
        print()
    finally:
        if speaker:
            speaker.close()


if __name__ == "__main__":
    main()
