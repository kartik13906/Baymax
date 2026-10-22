import re
import pandas as pd
from datasets import load_dataset

SAMPLE_SIZE = 3000
OUT_PATH = "clean_medical.jsonl"
MIN_WORDS, MAX_WORDS = 15, 250

def load_medquad(n):
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    df = ds.to_pandas().rename(columns={"Question":"question", "Answer": "answer"})
    df["source"] = "MedQuad"
    return df[["question", "answer", "source"]].sample(n=min(n, len(df)), random_state=42)

def load_chatdoctor(n):
    ds = load_dataset("lavita/chatDoctor-HealthCareMagic-100k", split="train")
    df = ds.to_pandas().rename(columns={"input": "question", "output": "answer"})
    df["source"] = "chatDoctor"
    return df[["question", "answer", "source"]].sample(n=min(n, len(df)), random_state=42)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
 
# ChatDoctor answers start with greetings and end with sign-offs.
GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|dear)[^.!?\n]{0,40}?(thanks? for[^.!?\n]*[.!?]|[,.!])\s*",
    re.IGNORECASE,
)
SIGNOFF_RE = re.compile(
    r"(take care|regards|best wishes|wish you|hope (this|it) helps|thanks?)[\s\S]{0,120}$",
    re.IGNORECASE,
)
DOCTOR_NAME_RE = re.compile(r"\bdr\.?\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?")
 
 
def clean_text(text: str) -> str:
    text = str(text)
    text = EMAIL_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = PHONE_RE.sub("", text)
    text = DOCTOR_NAME_RE.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)       # leftover HTML tags
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text
 
 
def strip_chatter(answer: str) -> str:
    """Remove 'Hi, thanks for your query' openers and 'Take care' closers."""
    answer = GREETING_RE.sub("", answer)
    answer = SIGNOFF_RE.sub("", answer)
    return answer.strip()
 
 
# ---------------------------------------------------------------
# 3. SAFETY FILTER
#    A 1.5B model must NOT learn to give drug doses. We DROP these
#    rows entirely instead of trying to edit them.
# ---------------------------------------------------------------
DOSE_RE = re.compile(
    r"\b\d+(\.\d+)?\s?(mg|mcg|µg|ml|iu|units?|tablets?|pills?|capsules?)\b",
    re.IGNORECASE,
)
 
# TODO (your turn #1): add more risky patterns, e.g. "prescribe",
# "antibiotic course", "twice daily", "every 8 hours". Be careful:
# too many patterns will throw away good data. Test on a sample.
RISKY_WORDS = ["prescribe", "twice daily", "three times a day", "every 8 hours"]
 
 
def is_risky(answer: str) -> bool:
    if DOSE_RE.search(answer):
        return True
    lowered = answer.lower()
    return any(w in lowered for w in RISKY_WORDS)
 
 
# ---------------------------------------------------------------
# 4. PUTTING IT TOGETHER
# ---------------------------------------------------------------
def main():
    df = pd.concat([load_medquad(SAMPLE_SIZE), load_chatdoctor(SAMPLE_SIZE)],
                   ignore_index=True)
    print(f"Loaded:            {len(df)} rows")
 
    df = df.dropna(subset=["question", "answer"])
 
    df["question"] = df["question"].map(clean_text)
    df["answer"] = df["answer"].map(clean_text).map(strip_chatter)
    print(f"After cleaning:    {len(df)} rows")
 
    # Length filter
    words = df["answer"].str.split().str.len()
    df = df[(words >= MIN_WORDS) & (words <= MAX_WORDS)]
    print(f"After length:      {len(df)} rows")
 
    # Safety filter
    before = len(df)
    df = df[~df["answer"].map(is_risky)]
    print(f"After safety:      {len(df)} rows (dropped {before - len(df)} risky)")
 
    # Deduplicate on lowercased question text
    df = df.drop_duplicates(subset=df["question"].str.lower().name)
    # TODO (your turn #2): also drop rows whose question is under 5 words.
 
    df.to_json(OUT_PATH, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(df)} rows -> {OUT_PATH}")
 
    # Always eyeball your data. Never trust a cleaning script blindly.
    print("\n--- 3 random samples ---")
    for _, row in df.sample(n=min(3, len(df)), random_state=1).iterrows():
        print(f"[{row.source}] Q: {row.question[:150]}")
        print(f"          A: {row.answer[:250]}\n")
 
 
if __name__ == "__main__":
    main()