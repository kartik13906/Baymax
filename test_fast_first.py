"""test_fast_first.py - must-pass tests for the optional fast first clause.   Run: python test_fast_first.py"""
import sys

from baymax_speak import speak_stream
from sentence_buffer import SentenceBuffer

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


def run(buf, pieces):
    out = []
    for p in pieces:
        out += buf.feed(p)
    return out + buf.flush()


TEXT = "That sounds uncomfortable, and I am sorry. Please rest, drink some water, and relax."


@check("default behaviour is unchanged: whole sentences only")
def _():
    assert run(SentenceBuffer(), [TEXT]) == ["That sounds uncomfortable, and I am sorry.", "Please rest, drink some water, and relax."]


@check("first_clause_only splits the first chunk at its comma, and only that one")
def _():
    out = run(SentenceBuffer(first_clause_only=True), [TEXT])
    assert out == ["That sounds uncomfortable,", "and I am sorry.", "Please rest, drink some water, and relax."], out


@check("a very short first clause is not split ('Hey, ...')")
def _():
    assert run(SentenceBuffer(first_clause_only=True), ["Hey, how is your day going?"]) == ["Hey, how is your day going?"]


@check("streaming one character at a time gives the same result")
def _():
    assert run(SentenceBuffer(first_clause_only=True), list(TEXT)) == run(SentenceBuffer(first_clause_only=True), [TEXT])


@check("model replies: the first clause is spoken early, and the guard still removes numbers in later parts")
def _():
    said = []
    stats = speak_stream(["Please rest your leg, and take 500 mg now. Drink water."], said.append, show=lambda s: None, fast_first=True)
    spoken = " ".join(said)
    assert "rest your leg" in spoken and "500" not in spoken, said
    assert any(r == "number" for r, _ in stats["removed"]), stats["removed"]


@check("an early first clause does not use up one of the five allowed sentences")
def _():
    reply = "That sounds uncomfortable, and I am sorry. " + " ".join(f"Sentence {w} is about rest." for w in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    said = []
    speak_stream([reply], said.append, show=lambda s: None, fast_first=True)
    assert len([s for s in said if "Sentence" in s]) == 4 and not any("zeta" in s or "epsilon" in s for s in said) and any("uncomfortable" in s for s in said), said


@check("fixed lines are never re-split, so they still match the pre-rendered audio")
def _():
    line = "If you are in danger, or have already hurt yourself, please call one one two right now."
    a, b = [], []
    speak_stream([line + " "], a.append, show=lambda s: None, guard=False, fast_first=True)
    speak_stream([line + " "], b.append, show=lambda s: None, guard=False, fast_first=False)
    assert a == b and len(a) == 1, a


print(f"\n{'all passed' if not failures else str(failures) + ' FAILED'}")
sys.exit(1 if failures else 0)
