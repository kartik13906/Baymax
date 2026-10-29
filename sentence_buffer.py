"""
sentence_buffer.py - Baymax Phase 3: the "Sentence Buffer" from your roadmap.

The language model streams its answer a few characters at a time ("I am", " sorry", " to"...).
Waiting for the whole answer before speaking would be slow, so this class collects the
pieces and hands back COMPLETE SENTENCES as soon as each one is finished. The voice can
start on sentence 1 while the model is still writing sentence 2.

    buf = SentenceBuffer()
    for piece in model_stream:
        for sentence in buf.feed(piece):
            speak(sentence)
    for sentence in buf.flush():        # whatever is left when the stream ends
        speak(sentence)

Splitting rules
  * after . ! ?  when followed by whitespace     ("3.5 mg" is not split, "Dr. Rao" is not split)
  * at a line break                              (list items, paragraphs)
  * optionally at commas (split_on_commas=True), only once the chunk is long enough
  * a bare list number like "1." is never a sentence on its own

first_clause_only=True is the compromise for speed: only the very first chunk may end at a comma (and only after
first_clause_words words). Each part is still checked by the guard on its own, but a removed second part can leave
the first part spoken, so baymax_speak.py uses it only for model replies, never for the fixed (cached) lines.

Why commas are OFF by default: your output guard judges whole sentences. If we speak the
first half of a sentence before the second half exists, the guard cannot catch a number or
a promise that appears later in that sentence. Turn commas on in Phase 5 only if
benchmarking shows you need the extra speed.
"""

import re

ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "vs", "st", "e.g", "i.e"}
LAST_WORD_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
LIST_NUMBER_RE = re.compile(r"\s*\d+")          # "1" before the "." of "1. Rest"


class SentenceBuffer:
    def __init__(self, split_on_commas=False, first_comma_words=4, later_comma_words=10,
                 first_clause_only=False, first_clause_words=3):
        self.buf = ""
        self.emitted = 0
        self.split_on_commas = split_on_commas
        self.first_comma_words = first_comma_words      # the first chunk may be short: speed matters most
        self.later_comma_words = later_comma_words      # later chunks wait longer: smoother speech
        self.first_clause_only = first_clause_only      # split at ONE comma, only for the very first chunk (faster start)
        self.first_clause_words = first_clause_words

    # ----- public -------------------------------------------------
    def feed(self, text):
        """Add streamed text. Returns a list of finished sentences (often empty)."""
        self.buf += text
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk, self.buf = self.buf[:cut], self.buf[cut:]
            chunk = self._clean(chunk)
            if chunk:
                out.append(chunk)
                self.emitted += 1
        return out

    def flush(self):
        """Call when the stream ends. Returns the leftover text as a final sentence."""
        chunk = self._clean(self.buf)
        self.buf = ""
        if chunk:
            self.emitted += 1
            return [chunk]
        return []

    # ----- internals ----------------------------------------------
    def _find_cut(self):
        buf = self.buf
        for i, ch in enumerate(buf):
            if ch == "\n":
                if self._has_letters(buf[:i]):
                    return i
                continue
            if ch not in ".!?,":
                continue
            # We only trust punctuation once we have seen the character AFTER it.
            # Otherwise "3." could still become "3.5" in the next piece of the stream.
            if i + 1 >= len(buf) or not buf[i + 1].isspace():
                continue
            if ch == ",":
                if self.first_clause_only and self.emitted == 0 and not self.split_on_commas:
                    if len(buf[:i].split()) >= self.first_clause_words:
                        return i + 1            # "That sounds uncomfortable," can be spoken while the rest is still being written
                    continue
                if not self.split_on_commas:
                    continue
                needed = self.first_comma_words if self.emitted == 0 else self.later_comma_words
                if len(buf[:i].split()) >= needed:
                    return i + 1
                continue
            if ch == "." and self._is_not_a_sentence_end(buf[:i]):
                continue
            return i + 1
        return None

    @staticmethod
    def _is_not_a_sentence_end(before_period):
        if LIST_NUMBER_RE.fullmatch(before_period):         # "1." at the start of a list item
            return True
        m = LAST_WORD_RE.search(before_period)
        if m:
            word = m.group(1).lower().rstrip(".")
            if word in ABBREVIATIONS:
                return True
            if len(word) == 1 and m.group(1).isupper():     # an initial, like "J. Smith"
                return True
        return False

    @staticmethod
    def _has_letters(text):
        return any(c.isalpha() for c in text)

    @classmethod
    def _clean(cls, text):
        text = re.sub(r"\s+", " ", text).strip()
        return text if cls._has_letters(text) else ""       # never speak a lone "1." or "-"
