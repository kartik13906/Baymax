"""
memory.py - Baymax's small, private memory (Phase 4: short-term memory in SQLite).

What it remembers, and ONLY what the person tells it:
    name         "my name is Kartik", "call me Kay"
    allergies    "I am allergic to peanuts and penicillin"
    pain log     every finished pain check-in: what happened, the 1-10 rating, and when

What you can say to it (while Baymax is awake):
    "what do you remember about me"        -> reads back name, allergies, last pain notes
    "how has my pain been"                 -> reads back the recent pain log
    "forget everything about me"           -> deletes the whole database content

Privacy rules (the tests check them):
  * The file stays on THIS computer (baymax_memory.db). Add it to .gitignore: it is personal health data.
  * Nothing here is ever sent to an online provider, and memory commands are not added to the chat
    history, so they cannot leak into a later casual message.
  * The model never sees the memory. Only fixed, pre-written sentences are built from it, so it cannot
    be twisted into advice. Allergies are NOT used to judge whether a remedy is safe: Baymax only repeats them.

This is a learning project, not a medical record. Do not rely on it for allergy safety.

Run  python memory.py  for a small self-test on a temporary in-memory database.
"""

import re
import sqlite3
import time

DEFAULT_PATH = "baymax_memory.db"
MAX_ALLERGIES = 12
NAME_STOPWORDS = {"later", "back", "maybe", "crazy", "tomorrow", "tonight", "now", "soon", "when", "if", "please",
                  "it", "that", "this", "a", "an", "the", "in", "at", "on", "up", "out", "sometime", "anytime"}

NAME_RE = re.compile(r"\b(?:my\s+name\s+is|my\s+name\'s|call\s+me|i\s+am\s+called)\s+([a-z][a-z\'-]{1,19})\b", re.I)
ALLERGIC_RE = re.compile(r"\b(?:i\s+am|i\'m|im)\s+(?:also\s+|really\s+|very\s+)?allergic\s+to\s+(.+)", re.I)
ALLERGY_OF_RE = re.compile(r"\bi\s+have\s+(?:an?\s+)?([a-z][a-z ]{1,30}?)\s+allerg(?:y|ies)\b", re.I)
RECALL_RE = re.compile(r"\bwhat\s+(?:do\s+you|can\s+you)\s+(?:remember|know)\s+about\s+me\b|\bwhat\s+do\s+you\s+remember\b", re.I)
PAIN_HISTORY_RE = re.compile(r"\b(?:how\s+(?:has|have|was|were)\s+my\s+pain|my\s+pain\s+(?:history|log)|pain\s+history)\b", re.I)
FORGET_RE = re.compile(r"\bforget\s+(?:everything|all|what\s+you\s+know)\b|\b(?:delete|erase|clear|wipe)\s+(?:all\s+)?(?:my|the)\s+(?:data|memory|memories|history|information)\b", re.I)


# Words that show a part of the sentence is NOT an allergen ("... and my throat is swelling").
NOT_AN_ALLERGEN = {"my", "i", "im", "me", "is", "am", "are", "was", "have", "has", "had", "feel", "feels", "hurts", "hurt",
                   "but", "because", "when", "since", "it", "its", "he", "she", "they", "we", "you", "not", "no"}


def _clean_item(text):
    """'peanuts.' / 'penicillin, please' -> 'peanuts' / 'penicillin'; rejects anything odd."""
    words = re.findall(r"[a-z][a-z\'-]*", text.lower())
    words = [w for w in words if w not in {"please", "really", "badly", "too", "also", "a", "an", "the", "some", "any"}]
    if not 1 <= len(words) <= 3 or any(w in NOT_AN_ALLERGEN for w in words):
        return None
    return " ".join(words)


def extract_name(text):
    m = NAME_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip("'-")
    if name.lower() in NAME_STOPWORDS:
        return None
    return name.capitalize()


def extract_allergies(text, with_dirty=False):
    """Allergens named in the text. with_dirty=True also says whether some other talk was mixed in
    ("allergic to cats and my nose hurts"), in which case Baymax should not swallow the whole message."""
    found, dirty = [], False
    m = ALLERGIC_RE.search(text)
    if m:
        chunk = re.split(r"[.!?;]|\bbut\b|\bso\b|\bwhich\b", m.group(1), maxsplit=1)[0]
        for part in re.split(r",|\band\b|\bor\b", chunk):
            item = _clean_item(part)
            if item:
                found.append(item)
            elif part.strip():
                dirty = True
    m = ALLERGY_OF_RE.search(text)
    if m:
        item = _clean_item(m.group(1))
        if item:
            found.append(item)
    found = list(dict.fromkeys(found))[:3]
    return (found, dirty) if with_dirty else found


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _when(ts, now):
    days = int(now // 86400 - ts // 86400)
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    return "a while ago"


class Memory:
    def __init__(self, path=DEFAULT_PATH, clock=time.time):
        self.db = sqlite3.connect(path)
        self.clock = clock
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS allergies (item TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS pain_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                                                 cause TEXT NOT NULL, rating INTEGER);
        """)
        self.db.commit()

    # -- storing ------------------------------------------------
    def set_name(self, name):
        self.db.execute("INSERT INTO facts(key, value) VALUES('name', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (name,))
        self.db.commit()

    def add_allergies(self, items):
        """Returns the items that were new."""
        have = set(self.allergies())
        new = [i for i in items if i not in have][: max(0, MAX_ALLERGIES - len(have))]
        for i in new:
            self.db.execute("INSERT OR IGNORE INTO allergies(item) VALUES(?)", (i,))
        self.db.commit()
        return new

    def log_pain(self, cause, rating):
        self.db.execute("INSERT INTO pain_log(ts, cause, rating) VALUES(?, ?, ?)",
                        (self.clock(), cause.strip()[:80], rating))
        self.db.commit()

    def forget_all(self):
        for table in ("facts", "allergies", "pain_log"):
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()
        self.db.execute("VACUUM")

    # -- reading ------------------------------------------------
    def name(self):
        row = self.db.execute("SELECT value FROM facts WHERE key='name'").fetchone()
        return row[0] if row else None

    def allergies(self):
        return [r[0] for r in self.db.execute("SELECT item FROM allergies ORDER BY item")]

    def recent_pain(self, n=3):
        return self.db.execute("SELECT ts, cause, rating FROM pain_log ORDER BY id DESC LIMIT ?", (n,)).fetchall()

    # -- talking ------------------------------------------------
    def learn(self, text):
        """Quietly store a name or allergy said in passing. Returns (name, new_allergies, mixed_with_other_talk)."""
        name = extract_name(text)
        if name:
            self.set_name(name)
        items, dirty = extract_allergies(text, with_dirty=True)
        return name, self.add_allergies(items), dirty

    def pain_summary(self):
        rows = self.recent_pain(3)
        if not rows:
            return "I have no pain notes yet."
        now = self.clock()
        parts = []
        for ts, cause, rating in rows:
            what = cause.rstrip(".")
            how = f", {rating} out of 10" if rating is not None else ""
            parts.append(f"{what}{how}, {_when(ts, now)}")
        return "Here are your latest pain notes. " + ". ".join(p[0].upper() + p[1:] for p in parts) + "."

    def respond(self, text):
        """If the message is about memory, store/forget/read back and return the fixed reply. Otherwise None."""
        if FORGET_RE.search(text):
            self.forget_all()
            return "Done. I have forgotten everything I knew about you."
        if PAIN_HISTORY_RE.search(text):
            return self.pain_summary()
        if RECALL_RE.search(text):
            bits = []
            if self.name():
                bits.append(f"Your name is {self.name()}.")
            if self.allergies():
                bits.append(f"You told me you are allergic to {_join(self.allergies())}.")
            if self.recent_pain(1):
                bits.append(self.pain_summary())
            return " ".join(bits) if bits else "I do not know anything about you yet. You can tell me your name."
        name, new, dirty = self.learn(text)
        if dirty:
            return None                                       # something else was said too: let Baymax answer the rest
        if new:
            return (f"Thank you for telling me, I noted {_join(new)}. "
                    "Please always tell your doctor or pharmacist about allergies yourself too.")
        if name:
            return f"Nice to meet you, {name}."
        return None


if __name__ == "__main__":
    m = Memory(":memory:")
    for line in ("my name is kartik", "I'm allergic to peanuts and penicillin.", "what do you remember about me?"):
        print("You:", line, "\nBaymax:", m.respond(line))
    m.log_pain("a heavy tin landed on my foot", 9)
    print("Baymax:", m.respond("how has my pain been"))
    print("Baymax:", m.respond("forget everything about me"))
    print("Baymax:", m.respond("what do you remember about me"))
