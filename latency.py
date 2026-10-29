"""
latency.py - measures where the time goes between "you stopped talking" and "Baymax makes a sound".

One Trace per turn holds timestamps (time.perf_counter seconds). Baymax's path has these marks, in order:

    end_of_speech   the VAD decided you finished (the 600 ms of silence it waited is NOT counted here)
    stt_done        Whisper finished writing your words
    first_token     the model produced its first piece of text      (missing for fixed replies)
    first_sentence  the first whole, guard-checked sentence was ready
    first_audio     the voice started playing

Each stage is the gap between two neighbouring marks that exist. A fixed reply (safety gate, remedy
line) has no first_token, so its "first sentence" stage runs straight from stt_done.

GOALS (edit them here): the roadmap's targets, measured from end_of_speech.
    first sound within 800 ms, and the voice itself (first sentence -> first sound) within 150 ms.
Add SILENCE_MS (600) on top to get what the person actually experiences after they stop talking.

Run  python latency.py  for a self-test.
"""

import statistics

GOAL_FIRST_AUDIO_MS = 800
GOAL_VOICE_MS = 150
SILENCE_MS = 600

ORDER = ["end_of_speech", "stt_done", "first_token", "first_sentence", "first_audio"]
LABELS = {"stt_done": "speech-to-text", "first_token": "model first token",
          "first_sentence": "first sentence ready", "first_audio": "voice first sound"}


class Trace:
    def __init__(self, **marks):
        self.marks = {k: v for k, v in marks.items() if v is not None}

    def mark(self, name, t):
        if t is not None:
            self.marks.setdefault(name, t)

    def stages(self):
        """[(label, ms), ...] for each stage that was measured."""
        marks = dict(self.marks)
        # A fixed line (for example the high-pain warning) can be spoken BEFORE the model answers. Then the
        # model's first token comes after the first sentence, so it is not part of the path to first sound.
        if "first_token" in marks and "first_sentence" in marks and marks["first_token"] > marks["first_sentence"]:
            del marks["first_token"]
        present = [n for n in ORDER if n in marks]
        return [(LABELS[b], (marks[b] - marks[a]) * 1000) for a, b in zip(present, present[1:])]

    def total_ms(self):
        if "end_of_speech" in self.marks and "first_audio" in self.marks:
            return (self.marks["first_audio"] - self.marks["end_of_speech"]) * 1000
        return None

    def voice_ms(self):
        if "first_sentence" in self.marks and "first_audio" in self.marks:
            return (self.marks["first_audio"] - self.marks["first_sentence"]) * 1000
        return None

    def line(self):
        parts = [f"{label} {ms:.0f} ms" for label, ms in self.stages()]
        total = self.total_ms()
        if total is None:
            return "   [latency: no sound was produced]"
        verdict = "within goal" if total <= GOAL_FIRST_AUDIO_MS else f"over the {GOAL_FIRST_AUDIO_MS} ms goal"
        return (f"   [latency: {' | '.join(parts)} | TOTAL {total:.0f} ms ({verdict}); "
                f"you also waited {SILENCE_MS} ms of silence before this]")


def percentile(values, p):
    """Nearest-rank percentile (p in 0-100). Simple and honest for small samples."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(p / 100 * len(ordered) + 0.5)) - 1))
    return ordered[k]


class Summary:
    """Collects many Traces and prints median / 95th percentile / worst per stage."""

    def __init__(self):
        self.stage_ms, self.totals, self.voices = {}, [], []

    def add(self, trace):
        for label, ms in trace.stages():
            self.stage_ms.setdefault(label, []).append(ms)
        if trace.total_ms() is not None:
            self.totals.append(trace.total_ms())
        if trace.voice_ms() is not None:
            self.voices.append(trace.voice_ms())

    def table(self):
        rows = [("stage", "median", "p95", "worst", "n")]
        for label in LABELS.values():
            v = self.stage_ms.get(label)
            if v:
                rows.append((label, f"{statistics.median(v):.0f}", f"{percentile(v, 95):.0f}", f"{max(v):.0f}", str(len(v))))
        if self.totals:
            t = self.totals
            rows.append(("TOTAL to first sound", f"{statistics.median(t):.0f}", f"{percentile(t, 95):.0f}", f"{max(t):.0f}", str(len(t))))
        widths = [max(len(r[i]) for r in rows) for i in range(5)]
        lines = ["  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(r)) for r in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines) + "  (ms)"

    def verdict(self):
        out = []
        if self.totals:
            med = statistics.median(self.totals)
            out.append(f"Median time to first sound: {med:.0f} ms. Goal {GOAL_FIRST_AUDIO_MS} ms -> "
                       f"{'MET' if med <= GOAL_FIRST_AUDIO_MS else 'NOT MET'}.")
        if self.voices:
            med = statistics.median(self.voices)
            out.append(f"Median voice stage (first sentence -> first sound): {med:.0f} ms. Goal {GOAL_VOICE_MS} ms -> "
                       f"{'MET' if med <= GOAL_VOICE_MS else 'NOT MET'}.")
        return "\n".join(out)


if __name__ == "__main__":
    t = Trace(end_of_speech=0.0, stt_done=0.20, first_token=0.55, first_sentence=0.70, first_audio=0.82)
    assert [round(ms) for _, ms in t.stages()] == [200, 350, 150, 120] and round(t.total_ms()) == 820
    fixed = Trace(end_of_speech=0.0, stt_done=0.2, first_sentence=0.21, first_audio=0.3)
    assert [l for l, _ in fixed.stages()] == ["speech-to-text", "first sentence ready", "voice first sound"]
    assert Trace(end_of_speech=0.0, stt_done=0.1).total_ms() is None
    warned = Trace(end_of_speech=0.0, stt_done=0.2, first_sentence=0.21, first_audio=0.3, first_token=0.9)
    assert all(ms >= 0 for _, ms in warned.stages()) and "model first token" not in [l for l, _ in warned.stages()]
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10 and percentile([], 50) is None
    s = Summary()
    s.add(t)
    s.add(fixed)
    print(t.line(), "\n")
    print(s.table(), "\n")
    print(s.verdict())
    print("\nlatency.py self-test passed")
