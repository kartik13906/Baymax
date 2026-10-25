# Baymax: a local, voice-first healthcare companion

> **Not medical advice.** This is a learning project. The language model is small (1.5B parameters) and **can state wrong facts with confidence**. The safety layer reduces risk but cannot remove it. Never use this software to make real medical decisions. In an emergency, call your local emergency number.

An offline healthcare companion with a gentle, calm personality. It was built to learn two things end to end: **fine-tuning a small language model** (QLoRA, on a 6 GB GPU) and **engineering a low-latency voice pipeline** (voice activity detection, speech-to-text, streaming text-to-speech). Everything runs on one machine, with no cloud APIs.

This is an independent fan project inspired by the fictional character Baymax. It is not affiliated with or endorsed by Disney.

## What it does

- **Persona model.** Qwen2.5-1.5B-Instruct fine-tuned with QLoRA to answer in short, kind sentences, ask for a 1-10 pain rating only when the person describes pain, suggest seeing a doctor, and avoid medicine names and doses.
- **Safety gate** (`safety_gate.py`). Plain rules, no language model. Checks the user's message first and replies with a fixed, pre-written answer for emergencies (chest pain, trouble breathing, stroke signs, heavy bleeding, and more), self-harm and crisis messages, and requests to diagnose a serious disease.
- **Output guard** (`output_guard.py`). Checks every sentence the model produces and drops those containing numbers, medicine names, schedules or promises it cannot keep (like "I will keep monitoring you"). It also removes markdown and emojis so the text sounds right when spoken.
- **Voice in.** Microphone, Silero VAD, then Faster-Whisper (`distil-medium.en`).
- **Voice out.** Kokoro-82M with adjustable pitch, softness and voice blending, fed by a sentence buffer so speech starts while the model is still writing.

## How a message flows

```
 microphone -> Silero VAD -> Faster-Whisper -> text
                                                |
                                       +--------v--------+
                                       | safety_gate.py  |-- emergency / crisis / diagnosis --> fixed reply (no model)
                                       +--------+--------+
                                                | clear
                                       Baymax model (Ollama, streaming)
                                                |
                              sentence_buffer.py -> output_guard.py (checks each sentence)
                                                |
                                          Kokoro voice -> speaker
```

`listen.py` (the input half) and `baymax_speak.py` (the output half) work today as separate pieces. Joining them into one asynchronous program is the next phase.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Environment setup | Done |
| 2 | Data cleaning, persona data, QLoRA training, export to Ollama | Done |
| 3 | Listening (VAD + Whisper) and speaking (sentence buffer + Kokoro) | Done as separate scripts |
| 4 | Safety gate and output guard | Done |
| 4 | Short-term memory (SQLite: name, allergies, recent pain ratings) | Not started |
| 5 | One async program, end-to-end latency benchmarks | Not started |

## Repository layout

| Stage | Files |
|---|---|
| Data cleaning | `clean.py` |
| Persona data | `persona.py`, `fix_pain.py`, `finalize_data.py` |
| Training | `prepare_dataset.py`, `train_qlora.py` |
| Evaluation and export | `test_adapter.py`, `merge_adapter.py`, `Modelfile` |
| Safety | `safety_gate.py`, `output_guard.py`, `test_safety.py`, `baymax_chat.py` |
| Wake phrases | `activation.py`, `baymax_live.py`, `test_activation.py` |
| Voice | `listen.py`, `sentence_buffer.py`, `speak.py`, `voice_lab.py`, `baymax_speak.py`, `test_speech.py` |

Trained adapters, GGUF files and downloaded models are **not** in the repository (they are large). The steps below rebuild them.

## Requirements

- Linux (developed on Arch). Python 3.10 or newer (developed on 3.14).
- An NVIDIA GPU with 6 GB of VRAM or more for training. Running the model and voice works on less.
- [Ollama](https://ollama.com) installed and running.
- Disk space: plan for roughly 15 GB (base model, merged model, GGUF, Whisper, Kokoro, caches).

## Setup

```bash
# system packages (Arch)
sudo pacman -S base-devel git portaudio ffmpeg

# virtual environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch: install the CUDA build that matches your driver (see pytorch.org)
pip install torch

# training and data
pip install transformers peft trl bitsandbytes datasets accelerate pandas numpy ollama

# listening
pip install faster-whisper silero-vad sounddevice

# speaking (see the note below about Python 3.14)
pip install kokoro-onnx espeakng-loader "phonemizer>=3.4.0"
```

**Python 3.14 note:** `kokoro-onnx` 0.6.1 declares that it needs Python below 3.14, so plain `pip install` quietly picks an old, incompatible version. On 3.14, install it like this instead:

```bash
pip install --no-deps --ignore-requires-python kokoro-onnx==0.6.1
pip install espeakng-loader "phonemizer>=3.4.0"
```

**Whisper on the GPU:** if you see `libcublas.so.12 is not found`, run `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`. `listen.py` finds those libraries by itself, and falls back to the CPU with a warning if the GPU still fails (or use `--stt-device cpu`).

## Build the model

Each step prints what it did. Run them in order from the project folder with the venv active.

1. **Clean the source data.** `python clean.py` downloads MedQuAD and ChatDoctor, removes personal details, greetings and sign-offs, drops answers with doses or prescribing language, and writes `clean_medical.jsonl`.
2. **Rewrite answers in the Baymax voice.** Run `ollama pull qwen2.5:7b`, then `python persona.py 1500` (hours; safe to stop and rerun, finished rows are skipped). Writes `baymax_data.jsonl`. Tip: run it with `nohup` and keep the machine awake.
3. **Fix the persona data.** `python fix_pain.py` removes the pain-rating question from rows where the person never mentioned pain, and thins out repeated openers. Writes `baymax_data_v2.jsonl`.
4. **Finalize.** `python finalize_data.py` drops answers with invented numbers or false promises and adds about 50 hand-written examples for dose requests, diagnosis requests, off-topic questions, crisis messages and emergencies. Writes `baymax_final.jsonl`.
5. **Format for training.** `python prepare_dataset.py` converts rows to ChatML with the Qwen tokenizer, repeats the hand-written behaviour examples three times, prints token-length statistics, and writes `train.jsonl` and `eval.jsonl`.
6. **Train.** Stop any model Ollama has loaded (`ollama ps`, then `ollama stop <name>`) so the GPU is free. Set `OUT_DIR` at the top of `train_qlora.py` (and make `output_dir=OUT_DIR` inside `SFTConfig`), then run `python train_qlora.py`. About 8 minutes on a 6 GB card. Settings: 4-bit NF4, LoRA rank 16, alpha 32, learning rate 2e-4, 2 epochs.
7. **Test the adapter.** `python test_adapter.py outputs/<your-run>` answers a set of prompts, including ones worded differently from the training data, with and without the adapter so you can see what changed.
8. **Export to Ollama.**
   ```bash
   python merge_adapter.py outputs/<your-run>
   git clone --depth 1 https://github.com/ggml-org/llama.cpp
   pip install gguf sentencepiece
   python llama.cpp/convert_hf_to_gguf.py outputs/baymax-merged --outfile outputs/baymax-q8.gguf --outtype q8_0
   ollama create baymax -f Modelfile
   ```

## Run it

| Command | What it does |
|---|---|
| `python baymax_chat.py` | Typed chat: safety gate, then the model, then the output guard. Shows what the gate and guard did. |
| `python baymax_live.py` | **Wake-phrase mode.** Baymax sleeps until it hears "ouch", "it hurts", "oh no", "I need help" or its name (or a safety-gate emergency). Then it asks for a pain rating and talks with you; "thank you" or silence puts it back to sleep. `--typed` tests it without a microphone, `--idle N` sets the sleep timeout. Add your own words in `EXTRA_TRIGGERS` in `activation.py`. |
| `python baymax_speak.py` | Same, but answers are spoken, and it prints timings (first token, first sentence, first sound). |
| `python listen.py` | Microphone to text, with timings and the safety gate's verdict. `--list-devices`, `--mic N`, `--stt-device cpu`. |
| `python speak.py "Hello"` | Speak one sentence. `--voice`, `--speed`, `--pitch`, `--softness`, `--save out.wav`. |
| `python voice_lab.py` | Renders candidate voices to `voice_lab/` so you can compare them. `--choose NAME` saves your pick to `voice.json`. |

The first run of the voice downloads the Kokoro model (about 350 MB) into `models/kokoro/`.

## The voice

Kokoro offers many built-in voices. `voice_lab.py` renders 8 voices (including two blends) in 3 characters each: `calm`, `deeper` and `big_soft`. Lower pitch makes a voice feel bigger, softness smooths harsh high frequencies, and blends mix two voices into a new one.

Cloning the recorded voice of a real person, or of a copyrighted character, is out of scope for this project. If you add voice cloning, use your own voice or one you have explicit permission to use.

## Safety design

| Layer | What it catches | What it cannot catch |
|---|---|---|
| **Safety gate** | Crisis and self-harm wording; cardiac, breathing, stroke, bleeding, unconscious, seizure, poisoning and allergy emergencies; "do I have cancer/TB/diabetes" style questions | Unusual phrasing, misspellings, other languages. It leans towards triggering when unsure. |
| **Output guard** | Numbers, medicine and test names, schedules ("every few hours"), promises, markdown, emojis, replies longer than 5 sentences | A wrong statement that contains none of those (for example, "store unopened insulin at room temperature") |
| **Model training** | Style, tone, refusing doses, redirecting off-topic questions | Medical knowledge. Facts come from the base model. |

The emergency number and the crisis helpline are settings at the top of `safety_gate.py` (`EMERGENCY_NUMBER`, `CRISIS_HELPLINE`). **Check them for your country before relying on them.**

## Tests

```bash
python test_safety.py   # 68 checks: messages that must trigger the gate, normal ones that must not, guard behaviour
python test_activation.py   # 15 checks: wake phrases, negations, sleep phrases, the awake/asleep session
python test_speech.py   # sentence buffer, speech queue, voice shaping, guard-in-the-stream (plus real synthesis if the Kokoro model is present)
```

When you find a message the gate should have caught (or wrongly caught), add it to `test_safety.py` first, then fix the rules until everything passes.

## Known limitations

- The model was trained on roughly 800 synthetic examples. It learned a **style**, not medicine.
- In early tests it gave wrong facts (for example, incorrect insulin storage advice) and falsely reassuring answers to serious-disease questions before the diagnosis rule existed. The gate and guard exist because of this.
- The gate is keyword-based and will miss things. Treat it as a safety net, not a guarantee.
- End-to-end latency has not been benchmarked yet.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Can't find 'adapter_config.json'` plus a Hugging Face 401 error | The adapter folder does not exist (training not finished, or saved elsewhere). `ls outputs` to check. A checkpoint folder works as an adapter too. |
| Out of memory while training | Set `per_device_train_batch_size=1` with `gradient_accumulation_steps=16`, lower `MAX_LEN` to 384, and free the GPU with `ollama stop <model>`. |
| `libcublas.so.12 is not found` | See "Whisper on the GPU" above. |
| `kokoro-onnx` complains about `colorlog` or `phonemizer-fork` | An old version was installed. Use the Python 3.14 commands above. |
| `SyntaxError` after pasting a file | A stray quote or a duplicated block. Run `python -m py_compile <file>`, or download the file again. |
| Nothing is heard by `listen.py` | `python listen.py --list-devices`, then `--mic N`. |

## Roadmap

- Join listening, safety, model and speaking in one asynchronous program (Phase 5).
- Measure every stage and the total time to first audio.
- SQLite short-term memory for name, allergies and recent pain ratings.
- Pre-render the emergency replies as audio files for instant playback.
- Try GPU text-to-speech if CPU speech is too slow.

## Data and licences

The training data comes from public medical Q&A sets (MedQuAD and ChatDoctor-HealthCareMagic) and was rewritten by a local language model. Check the terms of those datasets, and of the base model, before redistributing any data or trained weights. This repository has no licence file yet; add one before accepting contributions.

## Credits

[Qwen2.5](https://github.com/QwenLM/Qwen2.5), [PEFT](https://github.com/huggingface/peft) and [TRL](https://github.com/huggingface/trl), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggml-org/llama.cpp), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Silero VAD](https://github.com/snakers4/silero-vad), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) and [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx). See each project for its licence.
