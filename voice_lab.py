"""
voice_lab.py - find YOUR Baymax voice by ear.

It renders the same sentence in several Kokoro voices and several "characters"
(speed, pitch, softness) into the folder voice_lab/, one wav file each, so you can
listen and compare.

    python voice_lab.py                          # render everything into voice_lab/
    python voice_lab.py --play                   # also play each clip as it is rendered
    python voice_lab.py --text "Hello, I am Baymax."
    python voice_lab.py --choose am_michael__deeper    # save that choice to voice.json

After --choose, speak.py and baymax_speak.py use that voice automatically
(flags like --pitch 0.9 still override it).

What to listen for: Baymax is calm, slow, soft and a little synthetic, and sounds big and
gentle rather than sharp. Deeper pitch makes a voice feel bigger; softness takes the edge
off the "s" sounds; slower speed feels calmer. The "blend" rows mix two voices into one
that does not exist anywhere else. You can also write your own in the
VOICES dictionary below.
"""

import argparse
import json
import os
import time

from speak import DEFAULT_MODEL_DIR, VOICE_FILE, load_kokoro, make_synth, play_audio, save_wav

TEXT = ("I am Baymax, your personal healthcare companion. "
        "On a scale of 1 to 10, how would you rate your pain?")

# name -> Kokoro voice (or a blend "voice:weight,voice:weight")
VOICES = {
    "am_michael": "am_michael",
    "am_adam": "am_adam",
    "bm_george": "bm_george",
    "bm_lewis": "bm_lewis",
    "am_eric": "am_eric",
    "af_heart": "af_heart",
    "blend_michael_george": "am_michael:0.5,bm_george:0.5",
    "blend_adam_lewis": "am_adam:0.6,bm_lewis:0.4",
}

# name -> (speed, pitch, softness). Speed 1.0 is normal; pitch below 1.0 is deeper.
PRESETS = {
    "calm": (0.92, 1.00, 0.0),
    "deeper": (0.90, 0.92, 0.3),
    "big_soft": (0.88, 0.86, 0.6),
}


def all_candidates():
    for vname, spec in VOICES.items():
        for pname, (speed, pitch, softness) in PRESETS.items():
            yield f"{vname}__{pname}", spec, speed, pitch, softness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--out", default="voice_lab")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--play", action="store_true", help="play each clip after rendering it")
    ap.add_argument("--choose", metavar="NAME", help="save a candidate (e.g. am_michael__deeper) to voice.json")
    args = ap.parse_args()

    candidates = {name: (spec, speed, pitch, soft) for name, spec, speed, pitch, soft in all_candidates()}

    if args.choose:
        if args.choose not in candidates:
            raise SystemExit(f"No candidate named '{args.choose}'. Run without --choose to see the names.")
        spec, speed, pitch, soft = candidates[args.choose]
        with open(VOICE_FILE, "w", encoding="utf-8") as f:
            json.dump({"voice": spec, "speed": speed, "pitch": pitch, "softness": soft}, f, indent=2)
        print(f"Saved {VOICE_FILE}: voice={spec}, speed={speed}, pitch={pitch}, softness={soft}")
        print("speak.py and baymax_speak.py will use it from now on.")
        return

    os.makedirs(args.out, exist_ok=True)
    kokoro = load_kokoro(args.model_dir)
    print(f"Rendering {len(candidates)} clips into {args.out}/ ...\n")
    print(f"{'name':32s} {'speed':>5s} {'pitch':>5s} {'soft':>5s}   render time")
    for name, (spec, speed, pitch, soft) in candidates.items():
        synth = make_synth(kokoro, spec, speed, pitch, soft)
        t = time.perf_counter()
        audio, sample_rate = synth(args.text)
        took = (time.perf_counter() - t) * 1000
        save_wav(os.path.join(args.out, name + ".wav"), audio, sample_rate)
        print(f"{name:32s} {speed:5.2f} {pitch:5.2f} {soft:5.1f}   {took:6.0f} ms")
        if args.play:
            print(f"   playing {name} ...")
            play_audio(audio, sample_rate)

    print(f"\nListen to the files in {args.out}/, then pick one:")
    print("    python voice_lab.py --choose <name>        e.g.  --choose am_michael__deeper")


if __name__ == "__main__":
    main()
