"""
activation.py - decides WHEN Baymax wakes up and when it goes back to sleep.

Baymax stays asleep and ignores ordinary talk. It wakes up when it hears something that
sounds like pain or a call for help ("ouch", "it hurts", "my head hurts", "oh no",
"I need help", or its name). The safety gate (safety_gate.py) can also wake it: a
message about chest pain or self-harm always gets a reply, wake phrase or not.

    detector = TriggerDetector()
    t = detector.check("ouch, my arm hurts")      # -> Trigger(kind="pain", phrase="ouch")
    detector.check("the weather is nice")          # -> None

Edit the lists below to add your own words. Speech-to-text writes what it hears, so the
phrases are matched on normal lowercase text without punctuation.
"""

import re
from dataclasses import dataclass
from typing import Optional

from safety_gate import normalize

# Your own extra wake words (matched as whole words or phrases). Example: ["owie", "yikes"]
EXTRA_TRIGGERS = []

# (kind, pattern). The first match wins, so the most important kinds come first.
_PATTERNS = [
    ("pain", r"\b(ouch|ouchie|owie|ow)\b"),
    ("pain", r"\b(it|that|this)\s+(really\s+|so\s+|still\s+|hurts\s+)?hurts?\b"),
    ("pain", r"\b(my|the)\s+(\w+\s+){1,2}?(hurts?|hurting|aches?|aching|is\s+killing\s+me|is\s+sore|is\s+throbbing|is\s+burning)\b"),
    ("pain", r"\bi\s*(am|m|feel|have|got)\s+(in\s+)?(a\s+lot\s+of\s+|so\s+much\s+|some\s+)?(pain|hurt|hurting|injured)\b"),
    ("pain", r"\bi\s+(hurt|injured|burned|burnt|sprained|twisted|bruised|scraped)\s+(my|myself|the)\b"),
    ("pain", r"\bi\s+cut\s+(myself|my\s+(finger|hand|arm|leg|foot|toe|thumb|knee|lip|face))\b"),
    ("help", r"\b(i\s+need|please)\s+help\b|\bhelp\s+me\b"),
    ("distress", r"\boh\s+no\b"),
    ("name", r"\b(bay\s?max|bay\s?macks?|bay\s?mac)\b"),
]
_COMPILED = [(kind, re.compile(p)) for kind, p in _PATTERNS]

# Casual "I want your attention" phrases. They only count in SHORT utterances (6 words or
# fewer), so "hi" wakes Baymax but "I told hi to my friend and then we left" does not.
SHORT_MAX_WORDS = 6
_SHORT_PATTERNS = [
    ("greeting", r"^(hi|hii+|hello|hey|hiya|yo|namaste|good\s+(morning|afternoon|evening|night))\b"),
    ("attention", r"\bexcuse\s+me\b|\bare\s+you\s+(there|awake|listening|up)\b|\bcan\s+you\s+hear\s+me\b"
                  r"|\bwake\s+up\b|\b(talk|speak)\s+to\s+me\b|\bcan\s+we\s+(talk|chat)\b"
                  r"|\bi\s+(need|want)\s+(to\s+talk|someone|company|attention)\b|\bi\s*(am|m)\s+(bored|lonely)\b"),
]
_SHORT_COMPILED = [(kind, re.compile(p)) for kind, p in _SHORT_PATTERNS]

# "no chest pain", "it does not hurt": a negation right before, or inside, the match.
_NEGATION_BEFORE = re.compile(r"\b(no|not|never|without|dont|doesnt|didnt|stopped)\s+(longer\s+|any\s+|more\s+)?$")
_NEGATION_INSIDE = re.compile(r"\b(not|no|never|dont|doesnt|didnt|isnt|wasnt|cant|wont|stopped|without)\b")


@dataclass
class Trigger:
    kind: str       # "pain", "help", "distress", "name", "greeting", "attention" or "custom"
    phrase: str     # the words that matched


class TriggerDetector:
    def __init__(self, extra_phrases=None):
        extras = EXTRA_TRIGGERS if extra_phrases is None else extra_phrases
        self._extras = [re.compile(r"\b" + re.escape(normalize(p)) + r"\b") for p in extras if p.strip()]

    def check(self, text) -> Optional[Trigger]:
        norm = normalize(text)
        for kind, pattern in _COMPILED:
            for m in pattern.finditer(norm):
                if _NEGATION_BEFORE.search(norm[max(0, m.start() - 25):m.start()]):
                    continue
                if kind != "distress" and _NEGATION_INSIDE.search(m.group(0)):
                    continue
                return Trigger(kind, m.group(0))
        if len(norm.split()) <= SHORT_MAX_WORDS:
            for kind, pattern in _SHORT_COMPILED:
                m = pattern.search(norm)
                if m:
                    return Trigger(kind, m.group(0))
        for pattern in self._extras:
            m = pattern.search(norm)
            if m:
                return Trigger("custom", m.group(0))
        return None


# ---------------------------------------------------------------
# What Baymax says when it wakes up. Short, calm, and always a fixed line.
# ---------------------------------------------------------------
OPENINGS = {
    "pain": "I am here. On a scale of 1 to 10, how would you rate your pain?",
    "distress": "I heard that. Are you hurt? Please tell me what happened.",
    "help": "I am here. Please tell me what is wrong.",
    "name": "Hello. I am here. How are you feeling today?",
    "greeting": "Hello. I am Baymax. How are you feeling today?",
    "attention": "I am here. I am listening. What would you like to talk about?",
    "custom": "I am here. How can I help you?",
}
SLEEP_LINE = "Okay. I will rest now. Call me if you need me."


def opening_line(kind):
    return OPENINGS.get(kind, OPENINGS["name"])


# ---------------------------------------------------------------
# Going back to sleep
# ---------------------------------------------------------------
_SLEEP_RE = re.compile(
    r"\b(i\s*(am|m)\s+(satisfied|okay|ok|fine|good|better|all\s+good|all\s+better)(\s+now)?"
    r"|thank\s+you|thanks|that\s*(is|s)\s+all|goodbye|bye|go\s+to\s+sleep|never\s?mind|cancel|stop\s+listening)\b"
)
_FILLERS = {"you", "uh", "um", "hmm", "mm", "mhm", "huh", "the", "a", "and", "so"}


def is_sleep_phrase(text):
    """True for short messages like 'thank you' or 'I am satisfied'. A long message that
    merely contains 'thank you' is NOT a goodbye."""
    norm = normalize(text)
    return len(norm.split()) <= 6 and bool(_SLEEP_RE.search(norm))


def is_filler(text):
    """Noise that speech-to-text sometimes invents from silence ('you', 'um')."""
    words = normalize(text).split()
    return not words or (len(words) == 1 and (words[0] in _FILLERS or len(words[0]) <= 1))
