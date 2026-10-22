"""
persona_v2.py - Baymax Phase 2, step 2 (complete version)

Reads:   clean_medical.jsonl
Appends: baymax_data.jsonl   (safe to stop and rerun; finished rows are skipped)

Usage:
    python persona_v2.py 50      # timing/quality test
    python persona_v2.py 1500    # full run
"""

import json
import os
import random
import re
import sys
from collections import Counter

import ollama

MODEL = "qwen2.5:7b"
IN_PATH = "clean_medical.jsonl"
OUT_PATH = "baymax_data.jsonl"

# >>> If you tuned your own SYSTEM_PROMPT (extra examples etc.), paste it here
# >>> in place of this one.
SYSTEM_PROMPT = """You rewrite medical answers in the voice of Baymax, a gentle, caring healthcare companion.

Rules:
- Use exactly 3 or 4 short sentences. Never more.
- Only use facts stated in the original answer. Do not add new facts.
- When the person describes a symptom or pain, ask: "On a scale of 1 to 10, how would you rate your pain?"
- Only suggest safe first aid (rest, water, cold or warm compress). Never give drug names or doses.
- If the issue could be serious, say: "I would recommend seeing a doctor."
- No lists, no emojis, no markdown. Plain spoken sentences only.
- Output only the rewritten answer, nothing else.

Example
Question: I have had a headache for two days.
Original answer: Headaches can come from dehydration, stress or lack of sleep. Rest and drink fluids. See a doctor if it persists.
Baymax answer: I am sorry you are in discomfort. On a scale of 1 to 10, how would you rate your pain? Headaches often come from dehydration, stress or lack of sleep, so please rest and drink some water. If the pain continues, I would recommend seeing a doctor."""

DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|iu|units?|tablets?|pills?)\b", re.I)
BAD_PHRASES = ["as an ai", "i am an ai", "language model", "original answer", "baymax answer:"]


def rewrite(question: str, answer: str) -> str:
    short = " ".join(answer.split()[:80])          # cap the input length
    user_msg = f"Question: {question}\nOriginal answer: {short}\nBaymax answer:"
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.4, "num_predict": 130},
    )
    text = resp["message"]["content"].strip()
    if not text.endswith((".", "?", "!")) and "." in text:
        text = text.rsplit(".", 1)[0] + "."        # trim a cut-off sentence
    return text


def reject_reason(text: str):
    words = len(text.split())
    if words < 15:
        return "too short"
    if words > 120:
        return "too long"
    if DOSE_RE.search(text):
        return "dose"
    low = text.lower()
    for p in BAD_PHRASES:
        if p in low:
            return f"bad phrase: {p}"
    if "*" in text or "#" in text or "\n-" in text:
        return "markdown"
    return None


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    with open(IN_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.seed(42)
    random.shuffle(rows)                 # mix MedQuAD and ChatDoctor
    rows = rows[:limit]

    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            done = {json.loads(line)["question"] for line in f}

    todo = [r for r in rows if r["question"] not in done]
    print(f"Loaded {len(rows)} rows, {len(done)} already done, {len(todo)} to process.", flush=True)

    kept = 0
    reasons = Counter()

    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo, 1):
            try:
                text = rewrite(row["question"], row["answer"])
            except Exception as e:
                print(f"[{i}/{len(todo)}] error: {e}", flush=True)
                continue

            reason = reject_reason(text)
            if reason:
                reasons[reason] += 1
                print(f"[{i}/{len(todo)}] REJECTED ({reason})", flush=True)
                continue

            record = {
                "question": row["question"],
                "original_answer": row["answer"],
                "baymax_answer": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            kept += 1
            print(f"[{i}/{len(todo)}] kept so far: {kept}", flush=True)

    print(f"\nDone. Kept {kept}/{len(todo)} rows -> {OUT_PATH}")
    print("Rejection reasons:", dict(reasons))


if __name__ == "__main__":
    main()