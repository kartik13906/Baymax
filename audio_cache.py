"""
audio_cache.py - pre-rendered audio for the lines Baymax says word for word.

Why: the benchmark showed the voice needs about 0.5 s even for the short emergency line, because Kokoro
runs on the CPU. The fixed lines (safety-gate replies, wake-up lines, the pain questions, ...) never
change, so we synthesize them ONCE with  python prerender.py  and play them from memory afterwards.
An emergency reply then starts as soon as the gate fires.

How it plugs in: wrap(synth) returns a synth function with the same signature. If the exact text was
pre-rendered it returns the stored audio; otherwise it calls the real Kokoro voice. So nothing else
in the program changes, and a model sentence that happens to match a cached one is instant too.

The cache is tied to the voice settings (voice, speed, pitch, softness). Change them and the old files are
ignored until you run  python prerender.py  again. The folder is generated data: keep it out of git.
"""

import hashlib
import os

import numpy as np

CACHE_VERSION = 1          # bump when the file format changes


def voice_key(voice, speed, pitch, softness):
    return f"v{CACHE_VERSION}|{voice}|{float(speed):.3f}|{float(pitch):.3f}|{float(softness):.3f}"


class AudioCache:
    totals = {"hits": 0, "misses": 0}         # across all caches in this program (the benchmark prints them)

    def __init__(self, folder="audio_cache", key=""):
        self.folder, self.key = folder, key
        self.items = {}                       # spoken text -> (audio float32, sample rate)
        self.hits = self.misses = 0

    def _file(self, text):
        name = hashlib.sha1(f"{self.key}|{text}".encode()).hexdigest()[:20]
        return os.path.join(self.folder, name + ".npz")

    def put(self, text, audio, sample_rate):
        os.makedirs(self.folder, exist_ok=True)
        audio = np.asarray(audio, dtype=np.float32)
        np.savez_compressed(self._file(text), audio=audio, sr=np.array(sample_rate), text=np.array(text), key=np.array(self.key))
        self.items[text] = (audio, int(sample_rate))

    def load(self):
        """Read every file made for the CURRENT voice settings. Returns how many lines are ready."""
        self.items = {}
        if not os.path.isdir(self.folder):
            return 0
        for name in os.listdir(self.folder):
            if not name.endswith(".npz"):
                continue
            try:
                with np.load(os.path.join(self.folder, name)) as f:
                    if str(f["key"]) == self.key:
                        self.items[str(f["text"])] = (f["audio"], int(f["sr"]))
            except Exception:                  # a half-written or foreign file is just skipped
                continue
        return len(self.items)

    def has(self, text):
        return text in self.items

    def wrap(self, synth):
        def cached(text):
            hit = self.items.get(text)
            if hit is not None:
                self.hits += 1
                AudioCache.totals["hits"] += 1
                return hit
            self.misses += 1
            AudioCache.totals["misses"] += 1
            return synth(text)
        return cached

    def clean(self):
        """Delete every cache file (all voices). Returns how many were removed."""
        removed = 0
        if os.path.isdir(self.folder):
            for name in os.listdir(self.folder):
                if name.endswith(".npz"):
                    os.remove(os.path.join(self.folder, name))
                    removed += 1
        self.items = {}
        return removed


def fixed_lines():
    """Every line Baymax speaks word for word (whole messages; they are split into sentences later)."""
    from activation import ASK_RATING, ASK_RATING_AGAIN, HIGH_PAIN_LINE, OPENINGS, SLEEP_LINE
    from remedies import NO_REMEDY_LINE
    from safety_gate import DIAGNOSIS_MESSAGE, EMERGENCY_MESSAGES, EMERGENCY_NUMBER, _crisis_message
    lines = [m.format(EN=EMERGENCY_NUMBER) for m in EMERGENCY_MESSAGES.values()]
    lines += [_crisis_message(), DIAGNOSIS_MESSAGE, HIGH_PAIN_LINE, ASK_RATING, ASK_RATING_AGAIN, SLEEP_LINE, NO_REMEDY_LINE]
    lines += list(OPENINGS.values())
    return list(dict.fromkeys(lines))


def spoken_sentences(lines):
    """The exact strings the voice receives for these lines: split into sentences the same way the live loop
    does (SentenceBuffer), then made speakable (digits spelled out). Duplicates removed, order kept."""
    from baymax_speak import speak_stream
    heard = []
    for line in lines:
        speak_stream([line + " "], heard.append, show=lambda s: None, guard=False)
    return list(dict.fromkeys(heard))


def opener_sentences(replies, top=40, min_count=2, fast_first=True):
    """The first chunk the live loop would speak for each model reply, most common first: [(text, count), ...].

    Model replies start the same way again and again (the model was trained on a few dozen openers), and the first
    chunk is what the person waits for. Rendering the common ones ahead of time makes those replies start at once.
    It uses the real live path (guard on), so the strings match what the voice will receive. fast_first must match how you
    run Baymax: with --fast-first the first chunk is a clause, without it a whole sentence (prerender.py renders both)."""
    from collections import Counter
    from baymax_speak import speak_stream
    counts = Counter()
    for reply in replies:
        heard = []
        speak_stream([reply], heard.append, show=lambda s: None, guard=True, fast_first=fast_first)
        if heard:
            counts[heard[0]] += 1
    return [(t, n) for t, n in counts.most_common(top) if n >= min_count]


def training_replies(path="train.jsonl"):
    """The final assistant message of every training row (the thing the model learned to say)."""
    import json
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line).get("messages", [])
            if msgs and msgs[-1].get("role") == "assistant":
                out.append(msgs[-1]["content"])
    return out
