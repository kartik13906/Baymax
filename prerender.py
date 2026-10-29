"""
prerender.py - make the audio cache (see audio_cache.py).  Run it once, and again after you change the voice
or any fixed line (safety messages, wake-up lines, the pain questions).

    python prerender.py              # render what is missing for the current voice
    python prerender.py --list       # just show the sentences (no voice loaded)
    python prerender.py --clean      # delete the whole cache
    python prerender.py --voice af_heart --speed 1.0     # render for another voice setting

It also renders the most common OPENERS of the model's replies, found in train.jsonl (the first chunk of each reply
is what the person waits for). Use --top 0 to skip them, or --train PATH if the file lives elsewhere.
"""

import argparse
import time

import os

from audio_cache import AudioCache, fixed_lines, opener_sentences, spoken_sentences, training_replies, voice_key


def main():
    from speak import DEFAULT_MODEL_DIR, load_voice_settings
    s = load_voice_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="audio_cache")
    ap.add_argument("--voice", default=s["voice"])
    ap.add_argument("--speed", type=float, default=s["speed"])
    ap.add_argument("--pitch", type=float, default=s["pitch"])
    ap.add_argument("--softness", type=float, default=s["softness"])
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--train", default="train.jsonl", help="training file to mine for common reply openers")
    ap.add_argument("--top", type=int, default=40, help="how many common openers to render (0 = none)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()

    sentences = spoken_sentences(fixed_lines())
    openers = []
    if a.top > 0 and os.path.exists(a.train):
        replies = training_replies(a.train)
        openers = opener_sentences(replies, top=a.top, fast_first=True) + opener_sentences(replies, top=a.top, fast_first=False)
        sentences += [t for t in dict.fromkeys(t for t, _ in openers) if t not in sentences]
    elif a.top > 0:
        print(f"({a.train} not found: rendering fixed lines only. Run from the folder that has train.jsonl.)")
    if a.list:
        for t in sentences:
            print("  ", t)
        print(f"\n{len(sentences)} sentences ({len(set(t for t, _ in openers))} common openers from {a.train}, for both --fast-first and normal mode)")
        return
    cache = AudioCache(a.folder, voice_key(a.voice, a.speed, a.pitch, a.softness))
    if a.clean:
        print(f"Removed {cache.clean()} cached files.")
        return

    have = cache.load()
    todo = [t for t in sentences if not cache.has(t)]
    print(f"{len(sentences)} sentences: {len(sentences) - len(todo)} already cached ({have} files for this voice), {len(todo)} to render.")
    if not todo:
        return
    from speak import load_tts
    print(f"Loading the voice ({a.voice}) ...")
    synth, sr, _ = load_tts(a.model_dir, a.voice, a.speed, a.pitch, a.softness)
    seconds = 0.0
    t0 = time.perf_counter()
    for i, text in enumerate(todo, 1):
        audio, rate = synth(text)
        cache.put(text, audio, rate)
        seconds += len(audio) / rate
        print(f"  [{i}/{len(todo)}] {len(audio) / rate:4.1f} s  {text}")
    print(f"\nDone: {len(todo)} sentences, {seconds:.0f} s of audio, rendered in {time.perf_counter() - t0:.1f} s, saved to {a.folder}/")


if __name__ == "__main__":
    main()
