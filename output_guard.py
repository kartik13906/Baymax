"""
output_guard.py - Baymax Phase 4: clean the model's reply BEFORE it is shown or spoken.

Your tests showed the model inventing numbers ("68 to 77 degrees", "24 hours"),
promising things it cannot do ("I will continue monitoring you"), naming
medicines, and sometimes answering with markdown lists. This guard removes any
SENTENCE that does one of those things, and tidies the rest for text-to-speech.

It works per sentence on purpose: in Phase 3 the model streams tokens into a
sentence buffer, and you can call check_sentence() on each sentence before it
goes to the speech engine.

Usage:
    from output_guard import clean_reply, check_sentence
    reply, report = clean_reply(model_text)

LIMIT: this cannot catch a WRONG statement that contains no number, no drug and
no promise (for example "store unopened insulin at room temperature").
Rules catch patterns, not facts. That is why the gate and the disclaimers matter.
"""

import re

MAX_SENTENCES = 5     # short answers = lower speech latency
SAFE_FALLBACK = (
    "I am not able to give a reliable answer to that. "
    "I would recommend asking a doctor or a pharmacist."
)
DETAIL_NOTE = "For exact amounts, temperatures or timings, please check with a doctor or pharmacist."

# ---------------------------------------------------------------
# What gets a sentence dropped
# ---------------------------------------------------------------
# Numbers written as words. "one" is left out on purpose ("no one", "one of the").
NUMBER_WORDS = (
    "two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|eighteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    "hundred|thousand|twice|thrice|dozen"
)
NUMBER_WORD_RE = re.compile(rf"\b({NUMBER_WORDS})\b", re.I)
DIGIT_RE = re.compile(r"\d")
SCALE_RE = re.compile(r"\b1 to 10\b|\bone to ten\b", re.I)          # the pain question is allowed
FREQUENCY_RE = re.compile(
    r"\bevery\s+(\w+\s+)?(minute|minutes|hour|hours|day|days|night|nights|week|weeks|month|months)\b"
    r"|\b(hourly|daily|nightly|weekly|monthly)\b",
    re.I,
)

# Promises Baymax cannot keep.
PROMISE_RE = re.compile(
    r"\bI (will|shall|am going to|ll)\s+(continue|keep|monitor|check|remind|follow|track|watch|"
    r"stay|call|contact|notify|alert|send|schedule|book|be here|be watching|be monitoring)\b"
    r"|\bI(\'m| am) (monitoring|tracking|watching|checking)\b",
    re.I,
)

# A short list of common medicines. Not complete: the model can name a drug that
# is not here. CUSTOMIZE: add more as you spot them in your tests.
DRUG_WORDS = (
    "ibuprofen|paracetamol|acetaminophen|tylenol|aspirin|naproxen|diclofenac|"
    "antibiotic|antibiotics|amoxicillin|azithromycin|ciprofloxacin|penicillin|"
    "antihistamine|antidepressant|antacid|codeine|morphine|tramadol|prednisone|steroid|steroids|"
    "painkiller|painkillers|analgesic|analgesics|antitussive|decongestant|cetirizine|loratadine|"
    "metformin|insulin dose|opioid|opioids|sedative|sleeping pill|sleeping pills|melatonin|"
    "x-ray|xray|mri|ct scan|ultrasound"
)
DRUG_RE = re.compile(rf"\b({DRUG_WORDS})\b", re.I)

# ---------------------------------------------------------------
# Formatting that sounds wrong when spoken
# ---------------------------------------------------------------
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿️]")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)


def strip_formatting(text: str) -> str:
    text = EMOJI_RE.sub("", text)
    text = LIST_MARKER_RE.sub("", text)            # remove "1. ", "- ", "* " at line starts
    text = re.sub(r"[*_`#>]+", "", text)           # bold, italics, code, headings, quotes
    text = re.sub(r"\n+", ". ", text)              # line breaks become sentence breaks
    text = re.sub(r"\.\s*\.", ".", text)           # tidy ". ." produced above
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------
# The per-sentence check. Returns a reason string if the sentence must go.
# ---------------------------------------------------------------
def check_sentence(sentence: str):
    s = SCALE_RE.sub("", sentence)                 # allow "On a scale of 1 to 10 ..."
    if DIGIT_RE.search(s):
        return "number"
    if NUMBER_WORD_RE.search(s):
        return "number word"
    if FREQUENCY_RE.search(s):
        return "frequency"
    if PROMISE_RE.search(s):
        return "promise"
    if DRUG_RE.search(s):
        return "medicine or test"
    return None


def clean_reply(text: str):
    """Returns (cleaned_text, report). report lists what was removed and why."""
    report = {"removed": [], "used_fallback": False}

    sentences = split_sentences(strip_formatting(text))
    kept = []
    for s in sentences:
        reason = check_sentence(s)
        if reason:
            report["removed"].append((reason, s))
        else:
            kept.append(s)

    removed_detail = any(r in ("number", "number word", "frequency", "medicine or test")
                         for r, _ in report["removed"])

    kept = kept[:MAX_SENTENCES]

    if len(" ".join(kept).split()) < 6:            # almost nothing left
        report["used_fallback"] = True
        return SAFE_FALLBACK, report

    # Make the gap visible instead of silently leaving out a detail the person wanted.
    if removed_detail:
        if len(kept) >= MAX_SENTENCES:
            kept = kept[:MAX_SENTENCES - 1]
        kept.append(DETAIL_NOTE)

    cleaned = " ".join(kept)
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return cleaned, report


if __name__ == "__main__":
    import sys
    sample = " ".join(sys.argv[1:]) or input("Model reply to clean: ")
    out, rep = clean_reply(sample)
    print("CLEANED:", out)
    for reason, s in rep["removed"]:
        print(f"  removed ({reason}): {s}")
