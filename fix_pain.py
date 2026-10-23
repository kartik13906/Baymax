"""
fix_pain.py - clean up the persona data before training.

Fixes three things in baymax_data.jsonl:
  1. Removes the forced pain-scale question from rows where the person never
     mentioned pain or discomfort.
  2. Thins out the repeated "I am sorry to hear ..." opener, which would
     otherwise start most of the model's replies.
  3. Keeps spacing correct when a sentence is removed.

Reads:   baymax_data.jsonl   (always the ORIGINAL file, so it is safe to rerun)
Writes:  baymax_data_v2.jsonl

Run:     python fix_pain.py
"""

import json
import random
import re
from collections import Counter

IN_PATH, OUT_PATH = "baymax_data.jsonl", "baymax_data_v2.jsonl"

# Matches the whole sentence containing the pain-scale question.
PAIN_SENTENCE_RE = re.compile(r"[^.?!]*scale of 1 to 10[^.?!]*\?\s*", re.I)

# If the person's question has one of these, a pain question makes sense.
# CUSTOMIZE: add words you think signal pain or physical discomfort.
PAIN_WORDS_RE = re.compile(
    r"\b(pain|painful|hurt|hurts|hurting|ache|aches|aching|sore|soreness|"
    r"cramp|cramps|cramping|burning|stinging|discomfort|swollen|swelling|"
    r"headache|migraine|throbbing)\b",
    re.I,
)

# The repeated sympathy opener, e.g. "I am sorry to hear you are in discomfort."
SYMPATHY_RE = re.compile(r"^\s*I am sorry to hear[^.?!]*[.!]\s*", re.I)
KEEP_SYMPATHY_PROB = 0.5     # for rows where the person DID describe pain

# Detects sentences glued together like "that.The" (a missing space).
GLUED_RE = re.compile(r"[a-z][.?!][A-Z][a-z]")

MIN_WORDS = 12               # drop rows that become too short


def main():
    random.seed(42)
    with open(IN_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    kept, pain_removed, sympathy_removed, dropped = [], 0, 0, 0
    for r in rows:
        answer = r["baymax_answer"]
        has_pain = bool(PAIN_WORDS_RE.search(r["question"]))

        # 1. Remove the forced pain question. Replace it with ONE SPACE
        #    (not "") so the neighbouring sentences don't glue together.
        if not has_pain and PAIN_SENTENCE_RE.search(answer):
            answer = PAIN_SENTENCE_RE.sub(" ", answer)
            pain_removed += 1

        # 2. Thin out the sympathy opener: always remove it when the person
        #    did not describe pain, and remove it half the time otherwise.
        if SYMPATHY_RE.match(answer) and (not has_pain or random.random() > KEEP_SYMPATHY_PROB):
            answer = SYMPATHY_RE.sub(" ", answer, count=1)
            sympathy_removed += 1

        answer = re.sub(r"\s+", " ", answer).strip()
        if len(answer.split()) < MIN_WORDS:
            dropped += 1
            continue
        r["baymax_answer"] = answer
        kept.append(r)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(kept)
    with_pain = sum("scale of 1 to 10" in r["baymax_answer"].lower() for r in kept)
    glued = sum(bool(GLUED_RE.search(r["baymax_answer"])) for r in kept)
    print(f"Rows in:                  {len(rows)}")
    print(f"Pain question removed:    {pain_removed}")
    print(f"Sympathy opener removed:  {sympathy_removed}")
    print(f"Dropped (too short):      {dropped}")
    print(f"Rows out:                 {n}")
    print(f"Pain question share now:  {with_pain / n:.0%}")
    print(f"Glued sentences (want 0): {glued}")

    openers = Counter(" ".join(r["baymax_answer"].split()[:4]) for r in kept)
    print("\nMost common openers (first 4 words):")
    for text, count in openers.most_common(6):
        print(f"  {count:4d} ({count / n:.0%})  {text} ...")


if __name__ == "__main__":
    main()