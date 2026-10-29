"""
test_activation.py - must-pass tests for activation.py and the Session in baymax_live.py.
No microphone, speaker or model needed (fakes are used).

Run:   python test_activation.py
"""

import sys

from activation import TriggerDetector, is_filler, is_health_topic, is_sleep_phrase
from baymax_live import ACTIVE, FRIEND, SLEEPING, Session, api_messages
from remedies import ENTRIES, check_entries, find_remedy, with_remedy

results = []


def check(name):
    def wrap(fn):
        try:
            fn()
            results.append((True, name, ""))
        except Exception as e:
            results.append((False, name, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


D = TriggerDetector()

MUST_WAKE = [
    "ouch", "Ouch!", "ow", "Ow, that hurt", "oh no", "Oh no!", "oh, no", "it hurts", "It hurts!",
    "that hurts so much", "this really hurts", "my head hurts", "my left arm hurts a lot",
    "my back is killing me", "my knee is sore", "I am in pain", "I'm in pain", "I have a lot of pain",
    "I hurt my ankle", "I sprained my wrist", "I cut myself", "I cut my finger", "I burned my hand",
    "I need help", "please help", "help me", "hey baymax", "Baymax", "Hello Bay Max",
    "hi", "Hi!", "hello", "hey", "hey there", "hi there Baymax", "good morning", "Hello, are you there?",
    "excuse me", "can you hear me", "are you there", "can we talk", "I want to talk", "talk to me", "I am bored",
]
MUST_STAY_ASLEEP = [
    "the weather is nice today", "what time is it", "I went to the shop", "it does not hurt",
    "it doesn't hurt anymore", "my head does not hurt", "no pain at all", "I have no pain",
    "I am not in pain", "thanks for the book", "he said oh yes", "that was fun", "know what I mean",
    "I should know", "show me the way", "he helps me with homework",
    "I said hi to my neighbour and then we went inside", "she said hello to everyone at the party",
    "the history of this town is long", "this is what happened yesterday", "whiskey is not good",
]


@check("every wake phrase wakes Baymax")
def _():
    missed = [t for t in MUST_WAKE if D.check(t) is None]
    assert not missed, f"missed: {missed}"


@check("ordinary talk and negations do not wake Baymax")
def _():
    wrongly = [(t, D.check(t).phrase) for t in MUST_STAY_ASLEEP if D.check(t)]
    assert not wrongly, f"woke on: {wrongly}"


@check("trigger kinds: pain, distress, help, name")
def _():
    assert D.check("ouch").kind == "pain"
    assert D.check("oh no").kind == "distress"
    assert D.check("I need help").kind == "help"
    assert D.check("hey baymax").kind == "name"
    assert D.check("hi").kind == "greeting" and D.check("good evening").kind == "greeting"
    assert D.check("can you hear me").kind == "attention"


@check("your own extra wake words work")
def _():
    d = TriggerDetector(extra_phrases=["yikes", "Owie Zowie"])
    assert d.check("Yikes!").kind == "custom"
    assert d.check("owie zowie").kind in ("pain", "custom")
    assert d.check("yikesy") is None


@check("sleep phrases: short goodbyes yes, long sentences no")
def _():
    for t in ["thank you", "Thanks!", "I am satisfied now" if False else "I am fine now", "that is all", "never mind", "bye"]:
        assert is_sleep_phrase(t), t
    assert not is_sleep_phrase("thank you but my stomach still hurts a lot and I feel dizzy too")
    assert not is_sleep_phrase("seven")


@check("filler noise from silence is ignored")
def _():
    for t in ["", "you", "um", "uh", "hmm", "a"]:
        assert is_filler(t), t
    assert not is_filler("seven") and not is_filler("ouch")


HEALTH = [
    "i have a fever", "my stomach feels weird", "i feel dizzy", "which pills should i take", "i think i have the flu",
    "i've been feeling depressed", "my knee", "can you recommend a doctor", "i feel sick", "i have a rash on my arm",
    "i cut myself", "my blood pressure", "i'm having panic attacks", "i can't breathe well", "i feel kind of off today",
    "i have been coughing", "i think i'm allergic to peanuts", "what medicine is good for a cold", "i twisted my ankle",
]
NOT_HEALTH = [
    "hi", "hello baymax", "my day was so boring", "i'm craving something sweet", "tell me a fun fact", "it's cold today",
    "i got the job", "i'll be back later", "what's your favorite music", "i just finished a big project at work",
    "can we talk", "i'm so bored", "how was your day", "tell me a joke", "i went for a walk and it was lovely",
]


@check("health topics are recognised (they must stay on this computer)")
def _():
    missed = [t for t in HEALTH if not is_health_topic(t)]
    assert not missed, f"not recognised as health: {missed}"


@check("casual chat is not mistaken for health talk")
def _():
    wrong = [t for t in NOT_HEALTH if is_health_topic(t)]
    assert not wrong, f"wrongly health: {wrong}"


# ---------------------------------------------------------------
# Session with fakes
# ---------------------------------------------------------------
class Fake:
    def __init__(self, idle=30):
        self.now = 0.0
        self.fixed, self.replied, self.modes, self.skips = [], [], [], []
        self.s = Session(self.reply, self.fixed.append, idle_seconds=idle,
                         clock=lambda: self.now, log=lambda *_: None)

    def reply(self, text, history, mode="care", skip_scale=False):
        self.replied.append((text, history))
        self.modes.append(mode)
        self.skips.append(skip_scale)
        return ["Please rest and drink some water."]


@check("a remedy question with nothing on the list gets the fixed 'no reliable remedy' line, not the model")
def _():
    f = Fake()
    f.s.handle("hello baymax")
    for q in ("any tips for hiccups?", "how do i get rid of bad breath?"):
        assert f.s.handle(q) == "reply", q
        assert "reliable remedy" in f.fixed[-1], q
    assert f.replied == [], "the model must not be called for these"
    assert f.s.handle("any home remedy for a sore throat?") == "reply" and len(f.replied) == 1   # on the list: the model phrases it
    f.s.handle("hi again")
    assert f.s.handle("any tips for studying better?") == "reply"                              # not health: normal chat path


@check("'ouch' while already awake starts a new check-in instead of repeating the last answer")
def _():
    f = Fake()
    f.s.handle("hello baymax")
    f.s.handle("any home remedy for a sore throat?")
    n = len(f.replied)
    assert f.s.handle("ouch") == "intake"
    assert "where does it hurt" in f.fixed[-1].lower() and len(f.replied) == n
    assert f.s.handle("a heavy tin landed on my foot") == "intake" and "scale of 1 to 10" in f.fixed[-1]
    assert f.s.handle("9") == "reply" and "9 out of 10" in f.replied[-1][0]


@check("asleep: ordinary talk is ignored, nothing spoken, model unused")
def _():
    f = Fake()
    assert f.s.handle("the weather is nice") == "asleep"
    assert f.fixed == [] and f.replied == [] and f.s.state == SLEEPING


@check("'ouch' wakes Baymax and asks WHERE it hurts and what happened (not the rating yet)")
def _():
    f = Fake()
    assert f.s.handle("ouch") == "woke" and f.s.state == ACTIVE
    assert "where does it hurt" in f.fixed[0].lower() and "scale" not in f.fixed[0]


@check("'ouch my arm hurts' already says where, so Baymax goes straight to the rating question")
def _():
    f = Fake()
    assert f.s.handle("ouch my arm hurts") == "woke"
    assert "scale of 1 to 10" in f.fixed[0]
    assert f.s.handle("seven") == "reply"
    text, hist = f.replied[0]
    assert "my arm hurts" in text and "7 out of 10" in text and f.skips == [True]
    assert hist[0]["content"] == "ouch my arm hurts"


@check("full check-in: ouch -> what/where -> rating -> the model gets cause AND rating, and must not re-ask")
def _():
    f = Fake()
    f.s.handle("ouch")
    assert f.s.handle("i fell off my bike and scraped my knee") == "intake"
    assert "scale of 1 to 10" in f.fixed[-1] and f.replied == []
    assert f.s.handle("7") == "reply"                      # a bare digit must be heard (it used to be dropped)
    text, _hist = f.replied[0]
    assert "scraped my knee" in text and "7 out of 10" in text and f.skips == [True] and f.modes == ["care"]
    assert f.s.intake is None


@check("a rating given together with the cause skips the rating question")
def _():
    f = Fake()
    f.s.handle("ouch")
    assert f.s.handle("my wrist, about 6 out of 10") == "reply"
    assert "6 out of 10" in f.replied[0][0] and len(f.fixed) == 1


@check("every rating word and digit from 0 to 10 is understood")
def _():
    from activation import parse_pain_rating
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    for n, w in enumerate(words):
        assert parse_pain_rating(w) == n and parse_pain_rating(str(n)) == n, (n, w)
    assert parse_pain_rating("Seven.") == 7 and parse_pain_rating("about a six") == 6
    assert parse_pain_rating("8 out of 10") == 8 and parse_pain_rating("5/10") == 5
    assert parse_pain_rating("it hurts a lot") is None and parse_pain_rating("since two days", strict=True) is None


@check("no number heard: Baymax asks once more, then carries on without one")
def _():
    f = Fake()
    f.s.handle("ouch my arm hurts")
    assert f.s.handle("it is really bad") == "intake" and "did not catch a number" in f.fixed[-1]
    assert f.s.handle("it is really bad i said") == "reply"
    assert "out of 10" not in f.replied[0][0]


@check("a high rating adds a fixed 'get help' line before the model answers")
def _():
    f = Fake()
    f.s.handle("ouch my arm hurts")
    f.s.handle("nine")
    assert "medical help" in f.fixed[-1] and len(f.replied) == 1


@check("'thank you' in the middle of the check-in ends it cleanly")
def _():
    f = Fake()
    f.s.handle("ouch")
    assert f.s.handle("never mind") == "sleep" and f.s.intake is None and f.s.state == SLEEPING


@check("'i am so stressed' wakes Baymax and asks what is stressing you (friend mode, no pain question)")
def _():
    f = Fake()
    assert f.s.handle("i am so stressed") == "woke"
    assert "stressing you out" in f.fixed[0] and "scale" not in f.fixed[0]
    assert f.s.handle("my exams are next week") == "reply" and f.modes == ["friend"]


@check("the safety gate wakes Baymax even without a wake phrase, and the model is not used")
def _():
    f = Fake()
    assert f.s.handle("I have chest pain and it is spreading to my arm") == "gate"
    assert f.s.state == ACTIVE and f.replied == [] and len(f.fixed) == 1


@check("the safety gate still wins while active")
def _():
    f = Fake()
    f.s.handle("ouch")
    assert f.s.handle("I want to end my life") == "gate"
    assert f.replied == []


@check("'thank you' puts Baymax back to sleep and clears the history")
def _():
    f = Fake()
    f.s.handle("ouch my arm hurts")
    f.s.handle("seven")
    assert f.s.handle("thank you") == "sleep" and f.s.state == SLEEPING and f.s.history == []
    assert f.s.handle("seven") == "asleep"            # it no longer answers


@check("silence for longer than the idle time puts Baymax back to sleep")
def _():
    f = Fake(idle=30)
    f.s.handle("hi")
    f.now = 10
    assert f.s.handle("what a day") == "reply"        # still awake after 10 s
    f.now = 100
    assert f.s.handle("my stomach is upset") == "asleep" and f.s.state == SLEEPING


@check("noise like 'you' or 'um' never wakes or answers")
def _():
    f = Fake()
    assert f.s.handle("you") == "ignored"
    f.s.handle("ouch")
    assert f.s.handle("um") == "ignored" and f.replied == []


@check("history is capped at the last 6 messages")
def _():
    f = Fake()
    f.s.handle("hi")
    for i in range(10):
        f.s.handle(f"question number {i} about rest")
    assert len(f.s.history) == 6 and f.s.history[-2]["content"] == "question number 9 about rest"


@check("a greeting wakes Baymax and gets a casual hello, not a pain question")
def _():
    f = Fake()
    assert f.s.handle("hi") == "woke"
    assert "scale of 1 to 10" not in f.fixed[0] and "how is your day" in f.fixed[0].lower()
    assert f.s.handle("I am a bit tired today") == "reply"
    assert f.modes == ["friend"]


@check("friend mode switches to careful mode when pain is mentioned, and 'ouch' starts in careful mode")
def _():
    f = Fake()
    f.s.handle("hi")
    f.s.handle("what a day")
    f.s.handle("my back hurts a bit")
    assert f.modes == ["friend", "care"], f.modes
    g = Fake()
    g.s.handle("ouch my arm hurts")
    g.s.handle("seven")
    assert g.modes == ["care"]


@check("a health word in the middle of a friendly chat switches to care mode")
def _():
    f = Fake()
    f.s.handle("hi")
    f.s.handle("i feel a bit feverish today")
    assert f.modes == ["care"], f.modes


@check("a greeting that contains health talk wakes in care mode, not friend mode")
def _():
    f = Fake()
    f.s.handle("hi, i have a fever")
    f.s.handle("since monday")
    assert f.modes == ["care"], f.modes


@check("short follow-ups after health talk stay in care mode for a few turns, then friend mode returns")
def _():
    f = Fake()
    f.s.handle("hi")
    f.s.handle("my stomach hurts")          # care, hold = 3
    for t in ["since monday", "a bit worse", "no", "so what are you up to"]:
        f.s.handle(t)
    assert f.modes == ["care", "care", "care", "care", "friend"], f.modes


@check("what goes online contains ONLY friend-mode turns: no health talk can leak into a later casual message")
def _():
    f = Fake()
    f.s.handle("hi")
    f.s.handle("i have a bad headache")
    for t in ["it started yesterday", "ok", "yes", "tell me a joke"]:
        f.s.handle(t)
    assert f.modes[-1] == "friend"
    sent = api_messages("tell me a joke", f.s.history)
    text = " ".join(m["content"] for m in sent).lower()
    assert "headache" not in text and "yesterday" not in text, text
    assert sent[0]["role"] == "system" and sent[-1] == {"role": "user", "content": "tell me a joke"}
    assert all(set(m) == {"role", "content"} for m in sent)


@check("safety-gate replies are never part of what goes online")
def _():
    f = Fake()
    f.s.handle("hi")
    f.s.handle("i have chest pain and it spreads to my arm")
    f.s.handle("what a day")
    f.s.handle("tell me a joke")
    sent = " ".join(m["content"] for m in api_messages("x", f.s.history)).lower()
    assert "chest" not in sent and "emergency" not in sent, sent


@check("the remedy list is clean: guard-safe sentences, no medicines, every example finds its own entry")
def _():
    problems = check_entries()
    assert not problems, problems
    assert len(ENTRIES) >= 40


@check("remedy lookup: right entry for causes and questions, nothing for small talk")
def _():
    cases = {"i slammed my finger in the car door": "finger_door", "i bumped my head on a shelf": "head_bump",
             "any home remedy for a sore throat?": "sore_throat", "i burned my hand on the stove": "burn",
             "my period cramps are bad": "period", "i got a cramp in my leg": "cramp", "i have a nosebleed": "nosebleed",
             "hello there": None, "tell me a joke": None, "it's cold today": None, "what's your favorite food": None}
    for text, key in cases.items():
        found = find_remedy(text)
        assert (found["key"] if found else None) == key, (text, found and found["key"])


@check("the check-in hands the model the matching remedy, but history keeps only the plain message")
def _():
    f = Fake()
    f.s.handle("ouch")
    f.s.handle("i burned my hand on the stove")
    f.s.handle("six")
    sent = f.replied[0][0]
    assert "6 out of 10" in sent and "Remedies you may share" in sent and "cool running water" in sent, sent
    assert "Remedies you may share" not in " ".join(h["content"] for h in f.s.history), f.s.history


@check("a health question in care mode also gets its remedy; a casual one never does")
def _():
    f = Fake()
    f.s.handle("hi")
    f.s.handle("tell me a joke")
    f.s.handle("what can i do at home for a sore throat?")
    assert "Remedies you may share" not in f.replied[0][0]
    assert "Remedies you may share" in f.replied[1][0] and "pinch of salt" in f.replied[1][0]


@check("no remedy on the list: the message goes to the model unchanged (nothing invented)")
def _():
    f = Fake()
    f.s.handle("ouch")
    f.s.handle("my armpit hurts")
    f.s.handle("five")
    assert "Remedies you may share" not in f.replied[0][0]
    assert with_remedy("hello") == "hello"


failed = [r for r in results if not r[0]]
for ok, name, err in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n         {err}"))
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
