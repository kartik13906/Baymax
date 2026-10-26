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
    ("stress", r"\bi\s*(am|m|feel|have\s+been|been)\s+(so\s+|really\s+|very\s+|quite\s+)?(stressed|overwhelmed|anxious|worried|nervous|upset)\b"
               r"|\b(so|really)\s+stressed\b"),
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
    kind: str       # "pain", "help", "distress", "stress", "name", "greeting", "attention" or "custom"
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
# Is this message about health? Health talk NEVER goes to an online provider: it stays on this
# computer and is answered by the local care model. The list is deliberately broad: a false
# alarm only means a casual message is answered locally (safe); a miss would send health talk
# online (not safe). Add words freely.
# ---------------------------------------------------------------
_HEALTH_RE = re.compile(
    r"\b(fever|feverish|cough\w*|flu|headache|headaches|migraine\w*|nausea|nauseous|vomit\w*|diarrh\w+|dizzy|dizziness|"
    r"faint\w*|rash|itch\w*|swollen|swelling|bleed\w*|blood|infect\w*|allerg\w*|asthma\w*|diabet\w*|insulin|"
    r"pressure\s+is\s+high|cholesterol|cancer|tumou?r|pregnan\w*|injur\w*|wound\w*|bruis\w*|fractur\w*|sprain\w*|"
    r"sick|unwell|symptom\w*|medicine\w*|medication\w*|tablet\w*|pill|pills|dose|doses|dosage|antibiotic\w*|painkiller\w*|"
    r"doctor\w*|nurse|hospital\w*|clinic|pharmac\w+|diagnos\w+|disease\w*|treatment|surgery|vaccin\w+|prescri\w+|"
    r"depress\w+|anxiety|panic|suicid\w+|self\s?harm|hopeless|breath\w*|palpitation\w*|cramp\w*|constipat\w+|heartburn|"
    r"reflux|insomnia|fatigue|pain\w*|ache|aches|aching|hurt|hurts|hurting|sore|burning|numb\w*|tingling|weak|nauseated)\b"
    r"|\bmy\s+(head|stomach|tummy|chest|throat|back|neck|knee|knees|ankle|leg|legs|arm|arms|hand|hands|foot|feet|eye|eyes|"
    r"ear|ears|tooth|teeth|heart|skin|belly|joint|joints|shoulder|hip|wrist|finger|toe)\b"
    r"|\b(feel|feeling|felt)\s+(\w+\s+){0,2}(off|funny|strange|weird|terrible|awful|ill|rough|lousy|bad)\b"
    r"|\b(cut|burned|burnt|scraped|twisted|sprained)\s+(myself|my)\b"
)


def is_health_topic(text):
    """True when the message touches on health, symptoms, medicine, injuries or mental health."""
    return bool(_HEALTH_RE.search(normalize(text)))


# ---------------------------------------------------------------
# What Baymax says when it wakes up. Short, calm, and always a fixed line.
# ---------------------------------------------------------------
OPENINGS = {
    "pain": "I am here. Where does it hurt, and what happened?",
    "stress": "I am here. That sounds hard. What is stressing you out?",
    "distress": "I heard that. Are you hurt? Please tell me what happened.",
    "help": "I am here. Please tell me what is wrong.",
    "name": "Hey! I am here. How are you doing?",
    "greeting": "Hey! Good to hear from you. How is your day going?",
    "attention": "I am here, and I am all ears. What is on your mind?",
    "custom": "I am here. How can I help you?",
}
# The prompt for casual "friend" chat. prepare_dataset.py trains the friend examples with THIS exact
# text, so changing it means retraining.
FRIEND_SYSTEM = (
    "You are Baymax, a warm, relaxed friend who also cares about the person's health. "
    "Talk casually like a close friend: short natural sentences, contractions, a little humor, "
    "and ask a friendly question back. Reply in one to three sentences. "
    "Never name medicines, doses, numbers or diagnoses. "
    "If the person mentions pain or feeling unwell, be gentle and suggest seeing a doctor if it may be serious. "
    "Plain text only: no lists, no emojis."
)

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
    if not words:
        return True
    w = words[0]
    return len(words) == 1 and not w.isdigit() and (w in _FILLERS or len(w) <= 1)     # "7" is an answer, not noise


# ---------------------------------------------------------------
# The pain check-in: ask WHAT happened and WHERE, then HOW BAD (1 to 10).
# These questions are fixed lines (predictable and safe); the model answers afterwards.
# ---------------------------------------------------------------
ASK_RATING = "I am sorry you are hurting. On a scale of 1 to 10, how would you rate your pain?"
ASK_RATING_AGAIN = "Sorry, I did not catch a number. Please say a number from 1 to 10."
HIGH_PAIN_LINE = ("That sounds like a lot of pain. If it is very strong, sudden, or you feel faint, "
                  "please get medical help right away or ask someone nearby to help you.")
HIGH_PAIN_AT = 8
MIN_CAUSE_WORDS = 3          # "ouch my arm hurts" already says where; plain "ouch" does not

_NUMBER_WORDS = {"zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_WORD_RE = "|".join(_NUMBER_WORDS)
_EXPLICIT_RATING = re.compile(rf"\b(\d{{1,2}}|{_WORD_RE})\s*(out\s+of|over|/)\s*(10|ten)\b")
_ANY_NUMBER = re.compile(rf"\b(\d{{1,2}}|{_WORD_RE})\b")


def _to_int(token):
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


def parse_pain_rating(text, strict=False):
    """Find a pain rating from 0 to 10 in what the person said ("7", "seven", "about a six", "8 out of 10").
    strict=True only accepts the explicit forms "7 out of 10" / "7/10", so "since two days" is not a rating."""
    raw = re.search(r"\b(\d{1,2})\s*/\s*(10|ten)\b", text.lower())      # "5/10": normalize() would drop the slash
    if raw:
        n = int(raw.group(1))
        return n if 0 <= n <= 10 else None
    norm = normalize(text)
    m = _EXPLICIT_RATING.search(norm)
    if m:
        n = _to_int(m.group(1))
        return n if 0 <= n <= 10 else None
    if strict:
        return None
    if re.search(r"\bno\s+pain\b|\bnot\s+(really\s+)?(hurting|painful)\b", norm):
        return 0
    for m in _ANY_NUMBER.finditer(norm):
        n = _to_int(m.group(1))
        if 0 <= n <= 10:
            return n
    return None


def has_cause(text, trigger_phrase=""):
    """Does the wake-up message already say what hurts or what happened?"""
    rest = normalize(text).replace(normalize(trigger_phrase), " ") if trigger_phrase else normalize(text)
    return len(rest.split()) >= MIN_CAUSE_WORDS


def is_scale_question(sentence):
    """The model was trained to ask for a pain rating. Once we already have one, drop that sentence."""
    return bool(re.search(r"\bscale\s+of\s+1\s+to\s+10\b|\b(rate|rating)\b.*\bpain\b|\bpain\b.*\b(rate|rating)\b",
                          normalize(sentence)))
