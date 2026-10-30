"""
safety_gate.py - Baymax Phase 4: rule-based safety gate (NO LLM involved).

It checks the user's message BEFORE the language model sees it. If the message
describes an emergency, a self-harm crisis, or asks for a diagnosis of a serious
disease, it returns a fixed, pre-written reply and the LLM never runs.

Why rules and not the model? Your tests showed the 1.5B model answering
"my chest feels tight and I am sweating" with "rest and see a doctor".
Rules are predictable: the same message always gets the same safe answer.

Usage:
    from safety_gate import check_message
    result = check_message("My chest feels very tight")
    if result.triggered:
        speak(result.message)        # fixed reply, skip the LLM
    else:
        ...call the model...

Order of checks: crisis first, then emergency, then diagnosis.

IMPORTANT LIMITS: keyword rules miss unusual phrasings and misspellings. This is
a safety net for a learning project, not a medical device. Keep growing
test_safety.py every time you find a message the gate should have caught.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------
# SETTINGS - CHECK THESE FOR YOUR COUNTRY BEFORE YOU RELY ON THEM
# ---------------------------------------------------------------
# Override with environment variables (Docker does this); the defaults are India's numbers. Verify for your area.
EMERGENCY_NUMBER = os.environ.get("BAYMAX_EMERGENCY_NUMBER", "112")
CRISIS_HELPLINE = os.environ.get("BAYMAX_CRISIS_HELPLINE", "Tele-MANAS on 14416")    # "" = generic wording
# Note: these messages go to the text-to-speech engine later. Digits are fine for
# reading on screen, but you may want to spell them out for speech ("one one two").


# ---------------------------------------------------------------
# 1. NORMALISING THE TEXT
#    Lowercase, drop apostrophes ("can't" -> "cant"), turn other punctuation
#    into spaces, collapse whitespace. Patterns below are written against
#    this normalised form.
# ---------------------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------
# 2. PATTERNS
#    Each entry: (category, regex, negatable)
#    negatable=True  -> skip the match if it is negated, e.g. "no chest pain".
#    negatable=False -> the pattern already contains its own negation
#                       ("cant breathe", "wont stop bleeding"), so never skip it.
#    Crisis patterns are NEVER negation-checked: "I do not want to live" must trigger.
# ---------------------------------------------------------------
CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bsuicid\w*",
    r"\bend (my life|it all|everything)\b",
    r"\b(dont|do not) (want to|wanna) (live|be alive|be here|exist|go on)\b",
    r"\bwant to (die|disappear|be dead)\b",
    r"\b(wish|wished) (i|that i) (was|were|wasnt|werent) (dead|never born|not alive|not here)\b",
    r"\bwish i (could )?(disappear|die)\b",
    r"\b(no|not any) (reason|point) (to|in) (live|living|go on|going on|continue|continuing)\b",
    r"\bno point (in )?(going on|living|anything|continuing)\b",
    r"\b(dont|do not|cant|cannot) see (any|the|a) (point|reason) (in )?(going on|living|anything|continuing|life)\b",
    r"\bsee no (point|reason) (in )?(going on|living|anything|continuing)\b",
    r"\bnot worth living\b",
    r"\blife is not worth\b",
    r"\b(cant|cannot|can not) (go on|take it anymore|do this anymore)\b",
    r"\bbetter off (dead|without me)\b",
    r"\b(want|wanna|feel like|feeling like|thinking (of|about)|going to|plan(ning)? to|urge to|thoughts? of|tempted to)\s+(\w+\s+){0,2}?(hurt|hurting|harm|harming|kill|killing|cut|cutting)\s+myself\b",
    r"\bself harm\w*\b",
    r"\bhurt(ing)? myself on purpose\b",
]

# (category, regex, negatable)
EMERGENCY_PATTERNS = [
    # --- heart ---
    ("cardiac", r"\bchest\b.{0,25}\b(pain|pains|hurt|hurts|hurting|tight|tightness|pressure|crushing|squeez\w*|heavy|burning)\b", True),
    ("cardiac", r"\b(pain|pains|tightness|pressure|discomfort)\b.{0,15}\b(in|across|on|around)\b.{0,10}\bchest\b", True),
    ("cardiac", r"\bheart attack\b", True),
    ("cardiac", r"\bcardiac arrest\b", True),
    # --- breathing ---
    ("breathing", r"\b(cant|cannot|can not|unable to|struggling to|hard to|difficult to)\s+breathe?\b", False),
    ("breathing", r"\b(trouble|difficulty|problems?|struggling)\s+(with\s+)?breathing\b", True),
    ("breathing", r"\bshort(ness)? of breath\b", True),
    ("breathing", r"\b(not breathing|stopped breathing|stops breathing|gasping|choking)\b", False),
    # --- stroke ---
    ("stroke", r"\bface\b.{0,12}\b(drooping|droops|drooped|numb|dropping)\b", True),
    ("stroke", r"\bslurred speech\b", True),
    ("stroke", r"\b(cant|cannot|can not|unable to|suddenly cant)\s+speak\b", False),
    ("stroke", r"\bone side of (his|her|my|the|their)\s+(face|body)\b", True),
    ("stroke", r"\bsudden(ly)?\b.{0,20}\b(numbness|weakness|confusion)\b", True),
    ("stroke", r"\b(having|had|think .{0,10}) a stroke\b", True),
    # --- bleeding ---
    ("bleeding", r"\b(bleeding|bleed)\b.{0,15}\b(heavily|a lot|profusely|badly)\b", True),
    ("bleeding", r"\bheavy bleeding\b", True),
    ("bleeding", r"\b(wont|will not|cant|cannot|doesnt|does not|not)\s+stop(ping)?\s+bleeding\b", False),
    ("bleeding", r"\bbleeding\b.{0,12}\b(wont|will not|doesnt|does not|not)\s+stop\b", False),
    ("bleeding", r"\b(coughing|vomiting|throwing up|spitting) (up )?blood\b", True),
    ("bleeding", r"\blosing (a lot of )?blood\b", True),
    # --- unconscious ---
    ("unconscious", r"\b(unconscious|unresponsive|passed out|collapsed|fainted|not responding|not waking up|wont wake up|blacked out)\b", True),
    # --- seizure ---
    ("seizure", r"\b(seizure|seizures|convulsing|convulsion|convulsions)\b", True),
    # --- poisoning / overdose ---
    ("poison", r"\b(swallowed|drank|ate|eaten|ingested)\b.{0,25}\b(poison|poisonous|bleach|toxic|chemical|pesticide|detergent|battery)\b", True),
    ("poison", r"\bpoisoned\b", True),
    ("poison", r"\boverdos\w*\b", True),
    ("poison", r"\btook too (many|much)\s+(pills|tablets|capsules|meds|medicines?|medications?|sleeping|painkillers?|of (my|his|her|the|their))\b", True),
    # --- severe pain, about to faint, a broken bone, or asking for help ---
    ("severe", r"\b(unbearable|excruciating|agony|agonizing|agonising|unendurable)\b", True),
    ("severe", r"\b(pain|it|this|hurts?)\b.{0,20}\b(is|are|so|too|really)\b.{0,10}\b(unbearable|too much|too painful)\b", True),
    ("severe", r"\b(cant|cannot|can not)\s+(bear|stand|take|handle|stand)\s+(it|the pain|this|this pain)\b", False),
    ("severe", r"\b(about to|going to|gonna|feel like i will|feel like i might|think i will|think i am going to)\s+(faint|pass out|black out|collapse)\b", True),
    ("severe", r"\b(feel|feeling|am feeling)\s+(very |really |so )?faint\b", True),
    ("severe", r"\b(broken|fractured|snapped)\s+(bone|leg|arm|wrist|ankle|hip|back|neck|rib|ribs|collarbone)\b", True),
    ("severe", r"\b(leg|arm|wrist|ankle|hip|back|neck|rib|ribs|collarbone|bone)\b.{0,12}\b(is|looks|might be|may be)\s+(broken|fractured|snapped|bent the wrong way)\b", True),
    ("severe", r"\b(think|thinks) (i|he|she) (broke|snapped)\b", True),
    ("severe", r"\b(call|get|need)\s+(me\s+)?(an?\s+)?(ambulance|doctor|paramedic)\b", True),
    ("severe", r"^(please |somebody |someone |can you )?(help me|help|i need help|i need urgent help|please help)( now| please)?$", False),
    # --- severe allergic reaction ---
    ("allergy", r"\b(throat|tongue|lips?|face)\b.{0,15}\b(closing|swelling|swollen|swells)\b", True),
    ("allergy", r"\banaphyla\w*\b", True),
]

# Asking the bot to diagnose a serious disease. We never want the model to
# reassure ("a long cough is not usually TB") or to alarm.
SERIOUS_DISEASES = (
    r"(cancer|tumou?r|tuberculosis|tb|hiv|aids|diabetes|leukemia|leukaemia|"
    r"heart disease|kidney failure|liver failure|dementia|alzheimers|parkinsons|"
    r"multiple sclerosis|lupus|hepatitis)"
)
DIAGNOSIS_PATTERNS = [
    r"\b(do|did|could|might|may) (i|he|she|they) (have|has) (a |an |the )?(\w+\s+){0,2}" + SERIOUS_DISEASES + r"\b",
    # "is it cancer", "could this lump be a tumor", "could my cough be tuberculosis".
    # The lookahead requires the disease to END the question (or be followed by
    # "or", "because"...), so "is my mother's cancer curable" does NOT trigger.
    r"\b(is|could|might|can|may) ((it|this|that|these)( \w+){0,2}?|my( \w+){1,2}?) (be |possibly be |maybe be )?(a |an |the )?"
    + SERIOUS_DISEASES + r"(?=\s*$|\s+(or|and|because|since|if|given|after|when)\b)",
    r"\bam i (getting|developing|dying of|dying from) (a |an )?(\w+\s+){0,2}" + SERIOUS_DISEASES + r"\b",
    r"\b(diagnose|diagnosis)\b",
]

# Words that make a match "negated": "no chest pain", "without any bleeding".
NEGATION_BEFORE_RE = re.compile(
    r"\b(no|without|never|denies|dont have|do not have|doesnt have|does not have|"
    r"not having|nothing like|free of|is not|isnt|not that|not too|not so)\s+(any\s+|a\s+|the\s+|signs?\s+of\s+)?$"
)
# Negation INSIDE the matched span: "chest does not hurt", "no longer bleeding".
NEGATION_INSIDE_RE = re.compile(r"\b(not|no|never|dont|doesnt|didnt|isnt|wasnt|stopped)\b")


def _is_negated(norm: str, match: re.Match) -> bool:
    prefix = norm[max(0, match.start() - 30):match.start()]
    if NEGATION_BEFORE_RE.search(prefix):
        return True
    return bool(NEGATION_INSIDE_RE.search(match.group(0)))


# ---------------------------------------------------------------
# 3. THE FIXED REPLIES
#    Short, calm, Baymax-style, speakable (no lists, no markdown).
#    {EN} = emergency number. Each tells the person to call, then gives only
#    first-aid steps that are standard and safe.
# ---------------------------------------------------------------
def _crisis_message() -> str:
    where = f"{CRISIS_HELPLINE}, or a local crisis helpline" if CRISIS_HELPLINE else "a local crisis helpline"
    return (
        "I am really sorry you are in so much pain, and I am glad you told me. "
        "You matter, and you do not have to face this alone. "
        f"If you are in danger, or have already hurt yourself or taken anything, please call {EMERGENCY_NUMBER} right now. "
        f"You can also contact {where}. "
        "If you can, reach out to someone you trust and ask them to stay with you."
    )


EMERGENCY_MESSAGES = {
    "cardiac": (
        "This could be a heart emergency. Please call {EN} right now. "
        "Do not wait to see if it passes, and do not drive yourself. "
        "Sit down, stay as calm as you can, and ask someone nearby to stay with you."
    ),
    "breathing": (
        "Trouble breathing can be serious. Please call {EN} right now. "
        "Sit upright, try to stay calm, and ask someone to stay with you until help arrives."
    ),
    "stroke": (
        "These can be signs of a stroke, and every minute matters. Please call {EN} right now. "
        "Note the time the symptoms started, and do not give food or drink."
    ),
    "bleeding": (
        "Please press firmly on the wound with a clean cloth and keep pressing without lifting it. "
        "If the bleeding is heavy or does not slow down, call {EN} right now."
    ),
    "unconscious": (
        "Please call {EN} right now. "
        "If the person is breathing, gently place them on their side and stay with them. "
        "If they are not breathing, tell the emergency operator, and they will guide you."
    ),
    "seizure": (
        "Please call {EN} right now. "
        "Move hard objects away, put something soft under the head, and do not put anything in the mouth. "
        "Stay with the person until help arrives."
    ),
    "poison": (
        "Please call {EN} right now, or a poison control centre. "
        "Do not make the person vomit, and keep the container so you can tell the responders what was swallowed."
    ),
    "severe": (
        "That sounds serious, and I am worried about you. I cannot make calls myself, so please call {EN} now, "
        "or ask someone near you to call or take you to a hospital. "
        "Sit or lie down, keep still, and do not be alone until help arrives."
    ),
    "allergy": (
        "Swelling of the throat, tongue or face can be a severe allergic reaction. Please call {EN} right now. "
        "If the person has an adrenaline auto-injector, use it as directed on its label."
    ),
}

DIAGNOSIS_MESSAGE = (
    "I cannot diagnose a condition, because that needs a proper examination and sometimes tests. "
    "Many things can cause these symptoms, and I cannot tell which one this is. "
    "I would recommend seeing a doctor soon so they can check you properly."
)


# ---------------------------------------------------------------
# 4. THE GATE
# ---------------------------------------------------------------
@dataclass
class GateResult:
    triggered: bool
    category: Optional[str] = None     # "crisis", "cardiac", ..., "diagnosis"
    message: Optional[str] = None      # fixed reply to speak/print
    matched: Optional[str] = None      # the text that matched (for debugging)


def check_message(text: str) -> GateResult:
    norm = normalize(text)

    # 1. Crisis first: it must win even if the message also mentions pills etc.
    for pattern in CRISIS_PATTERNS:
        m = re.search(pattern, norm)
        if m:
            return GateResult(True, "crisis", _crisis_message(), m.group(0))

    # 2. Medical emergencies
    for category, pattern, negatable in EMERGENCY_PATTERNS:
        for m in re.finditer(pattern, norm):
            if negatable and _is_negated(norm, m):
                continue
            msg = EMERGENCY_MESSAGES[category].format(EN=EMERGENCY_NUMBER)
            return GateResult(True, category, msg, m.group(0))

    # 3. Requests to diagnose a serious disease
    for pattern in DIAGNOSIS_PATTERNS:
        m = re.search(pattern, norm)
        if m:
            return GateResult(True, "diagnosis", DIAGNOSIS_MESSAGE, m.group(0))

    return GateResult(False)


if __name__ == "__main__":
    # Quick manual try-out:  python safety_gate.py "my chest feels tight"
    import sys
    msg = " ".join(sys.argv[1:]) or input("Message: ")
    r = check_message(msg)
    print(f"triggered={r.triggered} category={r.category} matched={r.matched!r}")
    if r.message:
        print(r.message)
