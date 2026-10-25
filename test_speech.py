"""
test_speech.py - must-pass tests for sentence_buffer.py, speak.py (queue + helpers) and
the streaming pipeline in baymax_speak.py. No microphone, speaker or model needed,
except the optional real-voice check at the end (skipped if the model files are missing).

Run:   python test_speech.py
"""

import os
import sys
import time

import numpy as np

from baymax_speak import speak_stream
from output_guard import DETAIL_NOTE, SAFE_FALLBACK
from sentence_buffer import SentenceBuffer
from speak import (DEFAULT_MODEL_DIR, MODEL_FILE, VOICES_FILE, SpeechQueue, load_voice_settings,
                   resolve_voice, shape_voice, speakable)

results = []


def check(name):
    """Decorator: run the function, record PASS/FAIL (an assert or any error counts as FAIL)."""
    def wrap(fn):
        try:
            fn()
            results.append((True, name, ""))
        except Exception as e:
            results.append((False, name, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


def feed_all(buf, pieces):
    out = []
    for p in pieces:
        out += buf.feed(p)
    return out + buf.flush()


# ---------------------------------------------------------------
# SentenceBuffer
# ---------------------------------------------------------------
@check("splits at sentence ends across token boundaries")
def _():
    b = SentenceBuffer()
    assert b.feed("I am sorry to hear that. On a sc") == ["I am sorry to hear that."]
    assert b.feed("ale of 1 to 10, how would you rate your pain? Please rest.") == \
        ["On a scale of 1 to 10, how would you rate your pain?"]
    assert b.flush() == ["Please rest."]


@check("does not split inside a decimal number")
def _():
    out = feed_all(SentenceBuffer(), ["It is 3", ".", "5 degrees warmer. Next one."])
    assert out == ["It is 3.5 degrees warmer.", "Next one."], out


@check("does not split after Dr. or an initial")
def _():
    out = feed_all(SentenceBuffer(), ["Please see Dr. Rao today. Thank you."])
    assert out == ["Please see Dr. Rao today.", "Thank you."], out


@check("a bare list number is never a sentence on its own")
def _():
    out = feed_all(SentenceBuffer(), ["1. Rest well.\n2. Drink water.\n"])
    assert out == ["1. Rest well.", "2. Drink water."], out


@check("splits at line breaks")
def _():
    assert feed_all(SentenceBuffer(), ["Rest well\nDrink water\n"]) == ["Rest well", "Drink water"]


@check("commas are ignored by default")
def _():
    out = feed_all(SentenceBuffer(), ["Please rest, drink some water, and sleep. Thanks."])
    assert out == ["Please rest, drink some water, and sleep.", "Thanks."], out


@check("commas split once the chunk is long enough (when enabled)")
def _():
    out = feed_all(SentenceBuffer(split_on_commas=True), ["Please rest, drink some water, and sleep."])
    assert out == ["Please rest, drink some water,", "and sleep."], out


@check("character-by-character streaming gives the same result as one big piece")
def _():
    text = "I am sorry to hear that. It is 3.5 times more likely, says Dr. Rao. Please rest!\nThanks."
    assert feed_all(SentenceBuffer(), list(text)) == feed_all(SentenceBuffer(), [text])


@check("punctuation-only text produces nothing")
def _():
    assert feed_all(SentenceBuffer(), ["?! ... - \n"]) == []


# ---------------------------------------------------------------
# speakable
# ---------------------------------------------------------------
@check("long numbers are spelled digit by digit, small ones left alone")
def _():
    assert speakable("Call 112 now") == "Call one one two now"
    assert speakable("Tele-MANAS on 14416") == "Tele-MANAS on one four four one six"
    assert speakable("On a scale of 1 to 10") == "On a scale of 1 to 10"


# ---------------------------------------------------------------
# SpeechQueue (fake synthesizer and player)
# ---------------------------------------------------------------
def make_queue(fail_on=None):
    played = []

    def synth(text):
        time.sleep(0.02)
        if fail_on and fail_on in text:
            raise RuntimeError("synth failed on purpose")
        return np.zeros(10, dtype=np.float32), 24000, text

    def play(audio, sr, text):
        time.sleep(0.02)
        played.append(text)

    return SpeechQueue(synth, play), played


@check("speech queue plays sentences in order and wait() returns")
def _():
    q, played = make_queue()
    q.start_timer()
    for s in ["one", "two", "three", "four"]:
        q.say(s)
    q.wait()
    q.close()
    assert played == ["one", "two", "three", "four"], played
    assert q.ttfa_ms() is not None and q.ttfa_ms() > 0


@check("one failing sentence does not stop the rest or hang wait()")
def _():
    q, played = make_queue(fail_on="bad")
    for s in ["good one", "bad one", "good two"]:
        q.say(s)
    q.wait()
    q.close()
    assert played == ["good one", "good two"], played


@check("wait() with nothing queued returns immediately")
def _():
    q, _ = make_queue()
    t = time.perf_counter()
    q.wait()
    q.close()
    assert time.perf_counter() - t < 1


# ---------------------------------------------------------------
# speak_stream: buffer + guard together
# ---------------------------------------------------------------
def run(tokens, guard=True):
    said = []
    stats = speak_stream(tokens, said.append, show=lambda s: None, guard=guard)
    return said, stats


@check("clean reply is spoken in full, nothing removed")
def _():
    said, st = run(["I am sorry to hear that. ", "Please rest and drink some ", "water."])
    assert said == ["I am sorry to hear that.", "Please rest and drink some water."], said
    assert st["removed"] == []


@check("a sentence with a number is dropped and the doctor note is added at the end")
def _():
    said, st = run(["Rest is good. Store it between 68 and 77 degrees. Keep it out of sunlight."])
    assert said[:2] == ["Rest is good.", "Keep it out of sunlight."], said
    assert said[-1] == DETAIL_NOTE, said
    assert st["removed"][0][0] == "number"


@check("a promise is dropped")
def _():
    said, _ = run(["Back pain is common. I will continue monitoring your condition. Please rest."])
    assert said == ["Back pain is common.", "Please rest."], said


@check("if everything is removed, the safe fallback is spoken")
def _():
    said, _ = run(["Take 500 mg every 6 hours. Use 2 tablets daily."])
    assert said == [SAFE_FALLBACK], said


@check("markdown list items become plain sentences")
def _():
    said, _ = run(["Tips:\n1. Keep a regular bedtime.\n2. Avoid screens at night.\n"])
    assert "Keep a regular bedtime." in said and "Avoid screens at night." in said, said
    assert not any(s[0].isdigit() for s in said), said


@check("replies are cut off after 5 spoken sentences")
def _():
    text = " ".join(f"Sentence {w} is about resting." for w in
                    ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"])
    said, st = run([text])
    assert st["spoken"] == 5 and "zeta" not in " ".join(said), said


@check("fixed gate replies bypass the guard, and the phone number is spelled out for speech")
def _():
    msg = ["Please call 112 right now. Stay with the person. "]
    said, _ = run(msg, guard=False)
    assert said[0] == "Please call one one two right now.", said
    guarded, _ = run(msg, guard=True)                  # the same text WOULD be dropped by the guard
    assert "Please call one one two right now." not in guarded, guarded


# ---------------------------------------------------------------
# Voice shaping (pitch, softness, blending, voice.json)
# ---------------------------------------------------------------
def sine(freq, seconds=0.5, sr=24000, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@check("lower pitch gives a longer clip by 1/pitch and a lower frequency")
def _():
    x = sine(440)
    y = shape_voice(x, pitch=0.8)
    assert abs(len(y) - len(x) / 0.8) <= 1, (len(x), len(y))
    zero_crossings = lambda a: int(np.sum(np.diff(np.signbit(a).astype(int)) != 0)) / (len(a) / 24000)
    assert zero_crossings(y) < zero_crossings(x) * 0.9


@check("pitch 1.0 and softness 0 leave the audio unchanged")
def _():
    x = sine(300)
    assert np.allclose(shape_voice(x, 1.0, 0.0), x)


@check("softness removes high frequencies but keeps low ones")
def _():
    hi, lo = sine(7000), sine(200)
    rms = lambda a: float(np.sqrt(np.mean(a[200:-200] ** 2)))
    assert rms(shape_voice(hi, 1.0, 1.0)) < rms(hi) * 0.5
    assert rms(shape_voice(lo, 1.0, 1.0)) > rms(lo) * 0.9


@check("shaped audio never clips")
def _():
    assert float(np.abs(shape_voice(sine(300, amp=1.5), 0.9, 0.5)).max()) <= 0.99 + 1e-6


class FakeKokoro:
    def get_voices(self):
        return ["am_a", "bm_b"]

    def get_voice_style(self, name):
        return np.full((4, 1, 3), 1.0 if name == "am_a" else 3.0, dtype=np.float32)


@check("a plain voice name resolves to itself")
def _():
    assert resolve_voice(FakeKokoro(), "am_a") == "am_a"


@check("a blend is the weighted mix of the two voice styles")
def _():
    style = resolve_voice(FakeKokoro(), "am_a:0.75,bm_b:0.25")
    assert style.shape == (4, 1, 3) and np.allclose(style, 0.75 * 1.0 + 0.25 * 3.0), style


@check("an unknown voice stops with a clear message")
def _():
    try:
        resolve_voice(FakeKokoro(), "nope_voice")
    except SystemExit as e:
        assert "Unknown voice" in str(e)
    else:
        raise AssertionError("did not stop")


@check("voice.json: defaults, partial file, and a broken file")
def _():
    import json
    import tempfile
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            assert load_voice_settings()["pitch"] == 1.0                      # no file
            with open("voice.json", "w") as f:
                json.dump({"pitch": 0.9, "unknown_key": 5}, f)
            s = load_voice_settings()
            assert s["pitch"] == 0.9 and "unknown_key" not in s and s["speed"] > 0   # partial
            with open("voice.json", "w") as f:
                f.write("{ not json")
            assert load_voice_settings()["pitch"] == 1.0                      # broken file
        finally:
            os.chdir(old)


# ---------------------------------------------------------------
# Real voice (optional)
# ---------------------------------------------------------------
have_model = all(os.path.isfile(os.path.join(DEFAULT_MODEL_DIR, f)) for f in (MODEL_FILE, VOICES_FILE))
if have_model:
    @check("real Kokoro voice produces audible speech")
    def _():
        from speak import load_tts
        synth, sr, _names = load_tts()
        audio, sample_rate = synth("I am sorry to hear that.")
        assert sample_rate == 24000 and audio.dtype == np.float32
        assert len(audio) / sample_rate > 0.8, "audio is too short"
        assert float(np.abs(audio).max()) > 0.05, "audio is silent"

    @check("real voice: a deeper pitch keeps about the same speaking length, and a blend works")
    def _():
        from speak import load_kokoro, make_synth
        k = load_kokoro()
        text = "Please rest in a quiet room and drink some water."
        plain, sr = make_synth(k, "am_michael", 0.95, 1.0, 0.0)(text)
        deep, _ = make_synth(k, "am_michael", 0.95, 0.88, 0.5)(text)
        ratio = len(deep) / len(plain)
        assert 0.9 < ratio < 1.1, f"length ratio {ratio:.2f}: speed is not being compensated"
        mixed, _ = make_synth(k, "am_michael:0.5,bm_george:0.5", 0.95, 1.0, 0.0)(text)
        assert len(mixed) / sr > 0.8 and float(np.abs(mixed).max()) > 0.05
else:
    print(f"(skipping the real-voice check: model files not found in {DEFAULT_MODEL_DIR})")

# ---------------------------------------------------------------
failed = [r for r in results if not r[0]]
for ok, name, err in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n         {err}"))
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)