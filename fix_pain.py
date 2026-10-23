"""
fix_pain.py - remove the forced pain-scale question from rows where the
person never mentioned pain or discomfort.

Reads:   baymax_data.jsonl
Writes:  baymax_data_v2.jsonl

Run:     python fix_pain.py
"""

import json
import re
from collections import Counter

IN_PATH, OUT_PATH = "baymax_data.jsonl", "baymax_data_v2.jsonl"

# Matches the WHOLE sentence containing the pain-scale question, even if the
# wording varies ("...rate your pain?", "...rate your discomfort?").
PAIN_SENTENCE_RE = re.compile(r"[^.?!]*scale of 1 to 10[^.?!]*\?\s*", re.I)

# If the person's question contains one of these, the pain question makes sense.
# CUSTOMIZE: add words you think signal pain or physical discomfort.
PAIN_WORDS_RE = re.compile(
    r"\b(pain|painful|hurt|hurts|hurting|ache|aches|aching|sore|soreness|"
    r"cramp|cramps|cramping|burning|stinging|discomfort|swollen|swelling|"
    r"headache|migraine|throbbing)\b",
    re.I,
)

MIN_WORDS = 12   # drop rows that become too short after removal


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    kept, stripped, dropped = [], 0, 0
    for r in rows:
        answer = r["baymax_answer"]
        if not PAIN_WORDS_RE.search(r["question"]) and PAIN_SENTENCE_RE.search(answer):
            answer = PAIN_SENTENCE_RE.sub("", answer)
            answer = re.sub(r"\s+", " ", answer).strip()
            stripped += 1
        if len(answer.split()) < MIN_WORDS:
            dropped += 1
            continue
        r["baymax_answer"] = answer
        kept.append(r)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with_pain = sum("scale of 1 to 10" in r["baymax_answer"].lower() for r in kept)
    print(f"Rows in:                 {len(rows)}")
    print(f"Pain question removed:   {stripped}")
    print(f"Dropped (too short):     {dropped}")
    print(f"Rows out:                {len(kept)}")
    print(f"Pain question share now: {with_pain / len(kept):.0%}")

    # Check opener repetition: the model copies whatever starts most answers.
    openers = Counter(" ".join(r["baymax_answer"].split()[:6]) for r in kept)
    print("\nMost common openers:")
    for text, n in openers.most_common(5):
        print(f"  {n:4d}  {text} ...")


if __name__ == "__main__":
    main()