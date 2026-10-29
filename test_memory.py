"""
test_memory.py - must-pass tests for memory.py and its use in baymax_live.py.   Run: python test_memory.py
"""
import sys

from baymax_live import Session, api_messages
from memory import Memory, extract_allergies, extract_name

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


class Fake:
    def __init__(self):
        self.now = 1_000_000.0
        self.fixed, self.replied = [], []
        self.mem = Memory(":memory:", clock=lambda: self.now)
        self.s = Session(self.reply, self.fixed.append, clock=lambda: self.now, log=lambda *_: None, memory=self.mem)

    def reply(self, text, history, mode="care", skip_scale=False):
        self.replied.append(text)
        return ["Please rest and drink some water."]


@check("names are found, and look-alikes are not")
def _():
    assert extract_name("my name is kartik") == "Kartik" and extract_name("please call me Kay") == "Kay"
    assert extract_name("call me later") is None and extract_name("call me back please") is None


@check("allergies are found: lists, 'allergy' wording, trailing talk cut off")
def _():
    assert extract_allergies("I'm allergic to peanuts and penicillin.") == ["peanuts", "penicillin"]
    assert extract_allergies("i have a pollen allergy") == ["pollen"]
    assert extract_allergies("i am allergic to dust, but i am fine") == ["dust"]


@check("allergies mixed with other talk never swallow the message")
def _():
    f = Fake()
    f.s.handle("hello baymax")
    assert f.s.handle("i am allergic to cats and my nose hurts") != "memory"
    assert f.replied, "the rest of the sentence must still reach the model"


@check("saving and reading back: name, allergies, pain notes")
def _():
    f = Fake()
    f.s.handle("hello baymax")
    assert f.s.handle("my name is kartik") == "memory" and "Kartik" in f.fixed[-1]
    assert f.s.handle("i am allergic to peanuts") == "memory" and "peanuts" in f.fixed[-1]
    assert f.s.handle("ouch") == "intake"
    f.s.handle("a heavy tin landed on my foot")
    f.s.handle("9")
    f.now += 86400                                  # a day later Baymax has gone to sleep, so wake it again
    f.s.handle("hello baymax")
    assert f.s.handle("what do you remember about me") == "memory"
    said = f.fixed[-1]
    assert "Kartik" in said and "peanuts" in said and "heavy tin" in said and "9 out of 10" in said and "yesterday" in said, said


@check("the pain log records each finished check-in (rating may be missing)")
def _():
    f = Fake()
    f.s.handle("ouch my arm hurts")
    f.s.handle("seven")
    rows = f.mem.recent_pain(5)
    assert len(rows) == 1 and rows[0][2] == 7 and "arm" in rows[0][1]


@check("'forget everything' empties every table")
def _():
    f = Fake()
    f.s.handle("hello baymax")
    f.s.handle("my name is kartik")
    f.s.handle("i am allergic to peanuts")
    f.mem.log_pain("my knee", 4)
    assert f.s.handle("forget everything about me") == "memory"
    assert f.mem.name() is None and f.mem.allergies() == [] and f.mem.recent_pain() == []


@check("memory talk never enters the chat history, so it can never go online")
def _():
    f = Fake()
    f.s.handle("hi")
    f.s.handle("my name is kartik")
    f.s.handle("i am allergic to peanuts")
    f.s.handle("tell me a joke")
    sent = str(api_messages("how are you", f.s.history))
    assert "kartik" not in sent.lower() and "peanuts" not in sent.lower(), sent


@check("a known name personalises the greeting; no name, no change")
def _():
    f = Fake()
    f.mem.set_name("Kartik")
    f.s.handle("hello baymax")
    assert f.fixed[0].startswith("Hey Kartik!"), f.fixed[0]
    g = Fake()
    g.s.handle("hello baymax")
    assert g.fixed[0].startswith("Hey!"), g.fixed[0]
    h = Fake()
    h.s.handle("hello baymax my name is priya")
    assert h.mem.name() == "Priya"


@check("memory off (None) changes nothing")
def _():
    f = Fake()
    f.s.memory = None
    f.s.handle("hello baymax")
    assert f.s.handle("my name is kartik") == "reply"


print(f"\n{'all passed' if not failures else str(failures) + ' FAILED'}")
sys.exit(1 if failures else 0)
