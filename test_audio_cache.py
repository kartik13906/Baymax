"""test_audio_cache.py - must-pass tests for audio_cache.py (no voice model needed).   Run: python test_audio_cache.py"""
import sys
import tempfile

import numpy as np

from audio_cache import AudioCache, fixed_lines, spoken_sentences, voice_key

failures = 0


def check(name):
    def wrap(fn):
        global failures
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}   -> {e}")
        return fn
    return wrap


KEY = voice_key("am_adam", 1.0, 0.95, 0.2)


@check("a stored line comes back without calling the voice; anything else uses the voice")
def _():
    with tempfile.TemporaryDirectory() as d:
        c = AudioCache(d, KEY)
        c.put("I am here.", np.ones(100), 24000)
        calls = []
        synth = c.wrap(lambda t: calls.append(t) or (np.zeros(10), 24000))
        audio, sr = synth("I am here.")
        assert len(audio) == 100 and sr == 24000 and calls == []
        synth("something new")
        assert calls == ["something new"] and (c.hits, c.misses) == (1, 1)


@check("the cache survives a restart (files on disk)")
def _():
    with tempfile.TemporaryDirectory() as d:
        AudioCache(d, KEY).put("Call me if you need me.", np.arange(50), 24000)
        c2 = AudioCache(d, KEY)
        assert c2.load() == 1 and c2.has("Call me if you need me.")


@check("another voice setting ignores the old files, and the old voice still finds them")
def _():
    with tempfile.TemporaryDirectory() as d:
        AudioCache(d, KEY).put("Hey!", np.ones(5), 24000)
        assert AudioCache(d, voice_key("af_heart", 1.0, 1.0, 0.0)).load() == 0
        assert AudioCache(d, KEY).load() == 1


@check("a corrupt file is skipped, not fatal; clean() removes everything")
def _():
    with tempfile.TemporaryDirectory() as d:
        c = AudioCache(d, KEY)
        c.put("Okay.", np.ones(5), 24000)
        open(f"{d}/broken.npz", "wb").write(b"not a real file")
        assert AudioCache(d, KEY).load() == 1
        assert c.clean() == 2 and AudioCache(d, KEY).load() == 0


@check("every safety message is covered, and the sentences are exactly what the voice receives")
def _():
    sentences = spoken_sentences(fixed_lines())
    assert len(sentences) > 40 and len(sentences) == len(set(sentences))
    joined = " ".join(sentences)
    assert "one one two" in joined and "112" not in joined, "digits must be spelled the way the live voice gets them"
    from activation import ASK_RATING, HIGH_PAIN_LINE
    for line in (ASK_RATING, HIGH_PAIN_LINE):
        assert all(s in sentences for s in spoken_sentences([line]))
    from safety_gate import check_message
    for text in ("My chest feels very tight and I am sweating a lot.", "I want to end my life.", "Do I have cancer?",
                 "She collapsed and is not responding", "He is having a seizure", "I think I swallowed something poisonous."):
        gate = check_message(text)
        assert gate.triggered
        missing = [s for s in spoken_sentences([gate.message]) if s not in sentences]
        assert not missing, f"not cached for '{text}': {missing}"


@check("common openers are found the way the live loop speaks them, most common first, rare ones left out")
def _():
    from audio_cache import opener_sentences
    replies = (["That sounds uncomfortable, and I am sorry. Please rest."] * 3 + ["Hey! Good to hear from you."] * 2
               + ["Something one of a kind appears here. Done."])
    out = opener_sentences(replies, top=10, min_count=2)
    assert out[0] == ("That sounds uncomfortable,", 3), out        # the early first clause, as spoken with fast_first
    assert ("Hey!", 2) in out and all(n >= 2 for _, n in out) and len(out) == 2, out


@check("an opener that the guard would remove is never cached (it would never be spoken)")
def _():
    from audio_cache import opener_sentences
    out = opener_sentences(["Take 500 mg now. Please rest."] * 3, top=10, min_count=2)
    assert all("500" not in t for t, _ in out), out


@check("training_replies reads the last assistant message of each row")
def _():
    import json
    from audio_cache import training_replies
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/t.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]}) + "\n")
        assert training_replies(path) == ["Hello!"]


print(f"\n{'all passed' if not failures else str(failures) + ' FAILED'}")
sys.exit(1 if failures else 0)
