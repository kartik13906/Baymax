# Baymax: a local, voice-first healthcare companion

> **Not medical advice.** This is a learning project. The language model is small (1.5B parameters) and **can state wrong facts with confidence**. The safety layer reduces risk but cannot remove it. The home remedies are general first-aid and self-care steps written from general knowledge, **not reviewed by a clinician**. Never use this software to make real medical decisions. In an emergency, call your local emergency number.

A healthcare companion with a gentle, calm personality. It was built to learn two things end to end: **fine-tuning a small language model** (QLoRA, on a 6 GB GPU) and **engineering a low-latency voice pipeline** (voice activity detection, speech-to-text, streaming text-to-speech).

Everything about health runs on one machine. Casual "friend" chat can optionally use free online APIs (off by default); health talk, safety replies and your memory never leave the computer. See [Privacy](#privacy).

This is an independent fan project inspired by the fictional character Baymax. It is not affiliated with or endorsed by Disney.

## What it does

- **Persona model.** Qwen2.5-1.5B-Instruct fine-tuned with QLoRA to answer in short, kind sentences, suggest seeing a doctor, and avoid medicine names and doses.
- **Safety gate** (`safety_gate.py`). Plain rules, no language model. Checks the user's message first and replies with a fixed, pre-written answer for emergencies (chest pain, trouble breathing, stroke signs, heavy bleeding, and more), self-harm and crisis messages, and requests to diagnose a serious disease.
- **Output guard** (`output_guard.py`). Checks every sentence the model produces and drops those containing numbers, medicine names, schedules ("once a day") or promises it cannot keep. It also removes markdown and emojis so the text sounds right when spoken.
- **Wake phrases and two modes** (`activation.py`, `baymax_live.py`). Baymax sleeps until it hears a pain or help phrase ("ouch", "it hurts", "oh no", "I need help"), a greeting ("hi", "hey Baymax", "can we talk"), or the safety gate fires. Health talk runs in *care* mode on the local model; everything else runs in *friend* mode.
- **Pain check-in.** After "ouch" Baymax asks what happened and where, then for a 1 to 10 rating, then answers with both in hand. A rating of 8 or more first plays a fixed "get medical help" line. A bare "ouch" while awake starts a fresh check-in.
- **Home remedies, from a list** (`remedies.py`). 44 reviewable first-aid and self-care entries (cuts, burns, sprains, crush injuries, headache, sore throat and more). No medicines. The matching entry is attached to the model's message and the model only phrases it. If nothing matches, Baymax says it has no reliable remedy instead of inventing one (a fixed reply, no model call). `python remedies.py` validates the list.
- **Private memory** (`memory.py`). SQLite file on your computer with your name, allergies and recent pain check-ins. Ask "what do you remember about me", "how has my pain been" or say "forget everything about me".
- **Friend chat, optionally online** (`llm_backends.py`). With `--online-friend`, casual chat tries Groq, then OpenRouter, then falls back to the local model, with several models per provider and automatic cool-downs when a model is rate limited.
- **Voice in.** Microphone, Silero VAD, then Faster-Whisper (`distil-medium.en`) on the GPU.
- **Voice out.** XTTS v2 (a modern zero-shot cloner) speaks in a cloned voice from a few seconds of reference audio; Kokoro-82M (ONNX) remains available for fast preset voices with adjustable pitch, softness and blending. A sentence buffer starts speech while the model is still writing. Lines Baymax says word for word (safety replies, wake-up lines, the pain questions, common reply openers) are pre-rendered and play instantly, for the cloned voice as well as Kokoro.

## How a message flows

```
 microphone -> Silero VAD -> Faster-Whisper -> text
                                                |
                       memory commands <--------+
                                                |
                                       +--------v--------+
                                       | safety_gate.py  |-- emergency / crisis / diagnosis --> fixed reply (no model)
                                       +--------+--------+
                                                | clear
                     +--------------------------+--------------------------+
                     | health talk (care mode)                             | casual (friend mode)
              remedies.py attaches a remedy, or                     Groq -> OpenRouter -> local model
              a fixed "no reliable remedy" reply                    (only friend turns ever go online)
                     |
              Baymax model (Ollama, local, streaming)
                     |
        sentence_buffer.py -> output_guard.py (checks each sentence)
                     |
        audio cache (pre-rendered) or XTTS / Kokoro voice (GPU) -> speaker
```

## Speed

Measured with `python benchmark.py --runs 2 --voice-backend xtts` on this laptop (6 GB NVIDIA GPU; Whisper `distil-medium.en` and XTTS v2 on the GPU; the Q4_K_M model on the CPU because this Ollama build ships without a CUDA backend; 6 test sentences, 2 timed runs each). Time from "the voice activity detector decided you stopped talking" to "the first sound":

| Kind of turn | Default (median) | `--fast-first` (median) |
|---|---|---|
| Safety (gate) reply | 216 ms | 216 ms |
| Care reply | 218 ms | 221 ms |
| Friend chat | 1299 ms | 625 ms |
| **All test turns** | **688 ms** (goal 800 ms) | **299 ms** |
| p95 (slow tail) | 3597 ms | 1729 ms |

About 67% of the spoken chunks came from the pre-rendered audio cache in the default run. The slow tail is one long care sentence (a 27-word cold-vs-flu answer): the model takes ~0.9 s to finish that sentence and XTTS needs ~2.8 s to synthesize it. `--fast-first` speaks the first clause as soon as the model writes its first comma, which cuts that turn to ~1.7 s. The trade-off is unchanged: with `--fast-first` the output guard sees only the first clause, so a dangerous detail later in the same sentence can be removed after the clause has already been spoken. Keep it off if you want the strictest safety behavior.

The pure text path (`python baymax_speak.py --no-audio`) gets its first model token in **~0.1–0.2 s** when warm and its first sentence in **~0.7 s**. Before the Q4 model and the startup warm-up, the same first token took **~2.5 s** and the first sound **~3.4 s**.

Read these numbers with care: the test speech is synthetic and clean (speech-to-text is a best case), they do not include the 600 ms of silence Baymax waits to be sure you finished, or speaker buffering. `python baymax_live.py` prints a latency line after each real spoken turn. The earlier Kokoro benchmark (570 ms median) is not directly comparable: it did not run the cloned voice, and it had the model on the GPU.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Environment setup | Done |
| 2 | Data cleaning, persona data, QLoRA training, export to Ollama | Done (v7 model) |
| 3 | Listening (VAD + Whisper) and speaking (sentence buffer + XTTS / Kokoro) | Done |
| 4 | Safety gate and output guard | Done |
| 5 | Wake phrases, pain check-in, remedies, friend chat, SQLite memory | Done |
| 6 | One live program, latency benchmarks, pre-rendered audio, GPU voice | Done |
| 7 | Raspberry Pi client (microphone and speaker) talking to the laptop | Not started |

## Repository layout

| Stage | Files |
|---|---|
| Data cleaning | `clean.py` |
| Persona data | `persona.py`, `fix_pain.py`, `finalize_data.py`, `friend_data.py`, `extra_data.py` |
| Training | `prepare_dataset.py`, `train_qlora.py` |
| Evaluation and export | `test_adapter.py`, `test_adapter_paths.py`, `test_friend.py`, `test_checkin.py`, `merge_adapter.py`, `Modelfile` |
| Safety | `safety_gate.py`, `output_guard.py`, `test_safety.py`, `baymax_chat.py` |
| Wake phrases, modes, check-in | `activation.py`, `baymax_live.py`, `test_activation.py` |
| Remedies | `remedies.py` |
| Memory | `memory.py`, `test_memory.py` |
| Online friend chat | `llm_backends.py`, `test_backends.py`, `env.example` |
| Voice | `listen.py`, `sentence_buffer.py`, `speak.py`, `voice_lab.py`, `baymax_speak.py`, `test_speech.py`, `test_fast_first.py` |
| Web server and Docker | `server.py`, `web/index.html`, `Dockerfile`, `docker-compose.yml`, `docker-compose.gpu.yml`, `docker/entrypoint.sh`, `requirements-server.txt`, `test_server.py` |
| Speed | `latency.py`, `benchmark.py`, `audio_cache.py`, `prerender.py`, `test_audio_cache.py` |

Trained adapters, GGUF files, downloaded models, the memory database and the audio cache are **not** in the repository (large or personal). The steps below rebuild them.

## Requirements

- Linux (developed on Arch). Python 3.10 or newer (developed on 3.12; 3.14 needs the note below).
- An NVIDIA GPU with 6 GB of VRAM or more for training. Running the model, Whisper and the voice together fit in 6 GB.
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

# running Baymax (voice, listening, online chat)
pip install -r requirements.txt

# GPU voice (recommended): swap the CPU onnxruntime for the GPU build, see below
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -r requirements-gpu.txt

# only if you want to train the model yourself
pip install -r requirements-train.txt
```

**GPU voice.** `kokoro-onnx` pulls in the plain CPU `onnxruntime`. `onnxruntime` and `onnxruntime-gpu` use the same module name and must not be installed together, so uninstall both and install only the GPU build (`requirements-gpu.txt`: `onnxruntime-gpu` 1.26, a CUDA 12 build, with the CUDA libraries it needs from PyPI). On start the voice prints `Voice runs on: CUDAExecutionProvider`. If it says `CPUExecutionProvider` the GPU libraries did not load; everything still works, only slower. Set `BAYMAX_VOICE_CPU=1` to force the CPU. Skip the swap if you have no NVIDIA GPU.

**Whisper on the GPU:** if you see `libcublas.so.12 is not found`, run `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`. `listen.py` finds those libraries by itself and falls back to the CPU with a warning if the GPU still fails (or use `--stt-device cpu`).

**Python 3.14 note:** `kokoro-onnx` 0.6.1 declares that it needs Python below 3.14. On 3.14, install it with `pip install --no-deps --ignore-requires-python kokoro-onnx==0.6.1` and the other voice packages from `requirements.txt` separately.

## Build the model

Each step prints what it did. Run them in order from the project folder with the venv active.

1. **Clean the source data.** `python clean.py` downloads MedQuAD and ChatDoctor, removes personal details, greetings and sign-offs, drops answers with doses or prescribing language, and writes `clean_medical.jsonl`.
2. **Rewrite answers in the Baymax voice.** Run `ollama pull qwen2.5:7b`, then `python persona.py 1500` (hours; safe to stop and rerun, finished rows are skipped). Writes `baymax_data.jsonl`.
3. **Fix the persona data.** `python fix_pain.py` removes the pain-rating question from rows where the person never mentioned pain, and thins out repeated openers. Writes `baymax_data_v2.jsonl`.
4. **Finalize.** `python finalize_data.py` drops answers with invented numbers or false promises and adds about 50 hand-written examples for dose requests, diagnosis requests, off-topic questions, crisis messages and emergencies. Writes `baymax_final.jsonl`.
5. **Friend chat data.** `python friend_data.py` checks 155 hand-written casual conversations against the output guard and writes `baymax_friend.jsonl`.
6. **Check-in, remedy and multi-turn data.** `python extra_data.py` writes `baymax_extra.jsonl`: pain check-in conversations that replay the live bot's fixed lines, remedy rows built from `remedies.py`, "no reliable remedy" rows, multi-turn friend chats and more refusal wordings.
7. **Format for training.** `python prepare_dataset.py` converts all rows to ChatML with the Qwen tokenizer, trains friend rows under the friend system prompt (`activation.FRIEND_SYSTEM`), repeats the hand-written rows, prints token-length statistics, and writes `train.jsonl` and `eval.jsonl`.
8. **Train.** Stop any model Ollama has loaded (`ollama ps`, then `ollama stop <name>`) so the GPU is free. Set `OUT_DIR` at the top of `train_qlora.py` (and make `output_dir=OUT_DIR` inside `SFTConfig`), then run `python train_qlora.py`. About 15 minutes on a 6 GB card. Settings: 4-bit NF4, LoRA rank 16, alpha 32, learning rate 2e-4, 2 epochs. Use a new `OUT_DIR` for every run so you never overwrite an earlier adapter.
9. **Test the adapter.** `python test_adapter.py outputs/<run>` (care mode), `python test_friend.py outputs/<run>` (friend mode) and `python test_checkin.py outputs/<run>` (pain check-in, remedies, multi-turn chat). The automatic checks are only a rough filter, so read the replies yourself.
10. **Export to Ollama.**
    ```bash
    python merge_adapter.py outputs/<run>
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
    pip install gguf sentencepiece
    python llama.cpp/convert_hf_to_gguf.py outputs/baymax-merged --outfile outputs/baymax-q8.gguf --outtype q8_0
    ollama create baymax -f Modelfile
    ```
11. **Make a GPU-friendly build (recommended on a 6 GB card).** Ollama may quietly run the Q8 model on the CPU while Whisper and the voice hold the GPU, which made the first token take seconds. Convert the merged model to bf16, quantize to Q4_K_M, and register it as `baymax:q4`:
    ```bash
    python llama.cpp/convert_hf_to_gguf.py outputs/baymax-merged --outfile outputs/baymax-bf16.gguf --outtype bf16
    llama.cpp/build/bin/llama-quantize outputs/baymax-bf16.gguf outputs/baymax-q4.gguf Q4_K_M
    # Modelfile: FROM /absolute/path/outputs/baymax-q4.gguf
    ollama create baymax:q4 -f Modelfile
    ```
    `model_config.py` uses `baymax:q4` when it is installed and falls back to `baymax`, so the app keeps working on a machine that only has the original model. Set `BAYMAX_MODEL` to force a different tag, or `BAYMAX_KEEP_ALIVE` to change how long the model stays loaded (default `30m`). Both CLIs warm the model at startup, so the first real reply does not pay the load cost.

## Run it

| Command | What it does |
|---|---|
| `python baymax_live.py` | **The main program.** Microphone in, voice out. Sleeps until it hears a wake phrase. `--typed` tests it without a microphone, `--no-audio` skips the voice, `--idle N` sets the sleep timeout. |
| `python baymax_live.py --voice-backend xtts` | Speak in the cloned voice (the default). `--clone-voice NAME` picks another registered voice profile. |
| `python baymax_live.py --voice-backend voicekit` | Use the Voicekit / SV2TTS cloner instead of XTTS. `--voice-backend kokoro` goes back to the preset voices. |
| `python baymax_live.py --fast-first` | Start speaking after the first short clause of a model reply. Faster, with one trade-off: if the output guard removes the second half of that sentence, the first clause has already been spoken. |
| `python baymax_live.py --online-friend` | Casual chat goes to Groq, then OpenRouter (needs keys in `.env`; see below). |
| `python baymax_live.py --no-memory` | Remember nothing. `--memory PATH` changes where the memory file lives. |
| `python prerender.py` | Pre-render the fixed lines and the most common reply openers (mined from `train.jsonl`) into `audio_cache/`, for the cloned voice by default. Run once, and again after changing the voice or any fixed line. `--list` shows what it would render; `--voice-backend kokoro` renders the preset voice instead. |
| `python benchmark.py` | Time every stage with no microphone or speaker. `--fast-first`, `--no-cache`, `--online-friend`, `--stt-device cpu`. |
| `python baymax_chat.py` | Typed chat: safety gate, then the model, then the output guard. |
| `python baymax_speak.py` | Typed input, spoken answers, with timings. |
| `python listen.py` | Microphone to text, with timings and the safety gate's verdict. `--list-devices`, `--mic N`, `--stt-device cpu`. |
| `python speak.py "Hello"` | Speak one sentence. `--voice`, `--speed`, `--pitch`, `--softness`, `--save out.wav`. |
| `python voice_lab.py` | Renders candidate voices to `voice_lab/` so you can compare them. `--choose NAME` saves your pick to `voice.json`. |
| `python llm_backends.py --check` | Test your Groq and OpenRouter keys. `--models` lists the models your keys can use. |

The first run downloads the XTTS v2 model (about 1.8 GB) into the TTS cache. If you switch to `--voice-backend kokoro`, its model (about 350 MB) is downloaded into `models/kokoro/`.

### Things you can say

- **Wake up:** "ouch", "it hurts", "oh no", "I need help", "hi", "hey Baymax", "can we talk". Add your own in `EXTRA_TRIGGERS` in `activation.py`.
- **Remedies:** "any home remedy for a sore throat?" (answered from `remedies.py`).
- **Memory:** "my name is Kartik", "I am allergic to peanuts", "what do you remember about me", "how has my pain been", "forget everything about me".
- **Sleep:** "thank you", "that's all", or silence.

### Online friend chat

```bash
cp env.example .env        # then put your keys in .env (it is ignored by git)
python llm_backends.py --check
python llm_backends.py --models
python baymax_live.py      # keys found -> casual chat goes online automatically
```

If both keys are present, `baymax_live.py` turns online friend chat on by default. `--online-friend` still works and makes that explicit; `--local-friend` forces casual chat back to the local model. Wake greetings, safety replies and other fixed lines are spoken locally without calling any model, so the first "hello" is not an API call. Each casual model reply prints who answered it, for example:

```text
   [friend mode | answered by: groq | first token 375 ms]
```

The `.env` next to `baymax_live.py` is loaded even if you start the program from another folder. Restart the program after editing `.env`; a running process keeps the keys it loaded at startup.

Set `GROQ_MODELS` and `OPENROUTER_MODELS` in `.env` (comma separated, tried in order). Free tiers change often and have their own data policies: read them before sending anything you care about. In local testing the local friend model answered faster than the online providers; the online path is for variety and for taking load off the laptop.

## Run it with Docker (self-hosted)

Anyone can run Baymax on their own computer without installing Python, CUDA or the models by hand. It starts a web page; your browser provides the microphone and speaker, so it works on Windows, macOS, Linux and phones.

```bash
git clone https://github.com/kartik13906/Baymax.git && cd Baymax
cp env.example .env          # optional: API keys, emergency number, password
docker compose up --build    # first run downloads about 3 GB, then it is fast
```

Open **http://localhost:8000**, press **Start listening**, and say "hello Baymax" (or type it).

What starts, in order: `ollama` (runs the language model), `model-fetch` (copies `outputs/baymax-q4.gguf` if it exists, otherwise downloads it once), `model-init` (registers it with Ollama as `baymax`), and `baymax` (Whisper speech-to-text, the Kokoro voice, the safety layers and the web page). Stop with `docker compose down`; add `-v` to also delete the downloaded model and the memory file.

**If the model download returns 401.** The default Hugging Face repository is private until you publish it. Either put `outputs/baymax-q4.gguf` on the machine (the compose file copies it automatically), make the repository public, or set `HF_TOKEN` in `.env`. You can also point `BAYMAX_GGUF_URL` at any public `.gguf` URL. `BAYMAX_GGUF_SHA256` is optional but recommended for a shared file.

**Speed.** The default image runs everything on the CPU, so expect a few seconds per reply on an ordinary laptop. With an NVIDIA GPU and the NVIDIA Container Toolkit, run the language model on the GPU (the slowest part):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**Use it from a phone or another computer.** Browsers only allow the microphone on `https://` or `http://localhost`, so for other devices turn on the built-in self-signed certificate and set a password. In `.env`:

```
BAYMAX_BIND=0.0.0.0
BAYMAX_TLS=1
BAYMAX_TOKEN=pick-a-long-password
```

Then open `https://<your-computer's-address>:8000` and accept the certificate warning once. Never expose this to the public internet: it has no accounts, and everyone who can open the page shares one memory file. Set `BAYMAX_MEMORY=off` to remember nothing.

**Settings** (all in `.env`; see `docker-compose.yml`): `BAYMAX_EMERGENCY_NUMBER` (default 112) and `BAYMAX_CRISIS_HELPLINE` (default empty = generic wording). **Set these for where you live**: Baymax tells people to call this number in an emergency. `BAYMAX_WHISPER_MODEL` (`base.en` or `small.en` are faster on weak CPUs), `BAYMAX_MAX_SESSIONS`, `BAYMAX_IDLE` (seconds before it sleeps), `BAYMAX_LOCAL_FRIEND=1`, `GROQ_API_KEY` / `OPENROUTER_API_KEY` for the online friend chat.

**Share the built image (optional).** The model and Ollama stay separate, so publish only the `baymax` image:

```bash
docker build -t yourname/baymax:latest .
docker push yourname/baymax:latest
# on the other device:
BAYMAX_IMAGE=yourname/baymax:latest docker compose pull
BAYMAX_IMAGE=yourname/baymax:latest docker compose up --no-build
```

The other device still needs the model file (or `HF_TOKEN` / a public `BAYMAX_GGUF_URL`) and its own `.env`. Do not share `.env`: it contains API keys.

**Privacy in Docker.** Audio and text are processed inside your containers. Health talk, the pain check-in, remedies, memory and safety replies never leave them. Only casual chat can go to Groq/OpenRouter, and only if you add keys. Audio is not recorded to disk. If memory is on, the small `baymax-data` volume stores your name, allergies and pain notes; `docker compose down -v` deletes it. The image uses the preset Kokoro voice only; no cloned voice is included or supported in Docker.

### Publishing the model file (for the maintainer)

The fine-tuned model is too big for GitHub, so `docker-compose.yml` downloads it from a URL (`BAYMAX_GGUF_URL`). Once:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create baymax-gguf --type model
huggingface-cli upload kartik13906/baymax-gguf path/to/your-q4_k_m.gguf baymax-q4_k_m.gguf
sha256sum path/to/your-q4_k_m.gguf     # optional: put it in BAYMAX_GGUF_SHA256 so downloads are verified
```

Add a model card that says: fine-tuned from Qwen2.5-1.5B-Instruct (Apache-2.0), a learning project, **not a medical device**, and that it must be used with the safety layers in this repository (`safety_gate.py`, `output_guard.py`, `remedies.py`), because the raw model is not safe on its own. If your repository or file name differs from the default, set `BAYMAX_GGUF_URL` in `.env`.

### Before you share this publicly

- Have a clinician or a trusted first-aid source review `remedies.py` (44 entries). The code stops the model inventing remedies; it cannot check that the listed ones are right.
- Check the emergency number and helpline for the countries you expect users in. The defaults are India's.
- Keep the "not a medical device" notice (it is on the page and in the model card). Baymax gives general comfort and first-aid style tips, never diagnoses or medicines.
- Free API tiers have their own data policies; read them before enabling the online friend chat for other people.

## Privacy

- **Never leaves the computer:** health talk, pain check-ins, safety replies, remedies, your name, allergies and pain log. Each turn is tagged with its mode, and only friend-mode turns can be built into an online request (`api_messages` in `baymax_live.py`). Wake greetings, fixed lines and memory commands are spoken locally and are never added to an online request.
- **Online (only for friend-mode chat):** casual friend-mode messages go to Groq or OpenRouter when keys are present, unless you start with `--local-friend`.
- **Files to keep out of git:** `.env` (API keys), `baymax_memory.db` (your personal health notes), `audio_cache/`, `outputs/`, `models/`. The included `.gitignore` covers them.

## The voice

**Cloned voice (default).** `baymax_live.py` and `baymax_speak.py` speak with **XTTS v2**, loaded once at startup. The reference clips live in `~/.voicekit/voices/<name>/xtts_refs/*.wav`; the cloner uses the first 6 seconds for prosody and averages the speaker embedding over all of them. Ten to thirty seconds of clean, single-speaker audio works best. `BAYMAX_XTTS_REFS=/path/a.wav:/path/b.wav` overrides the clips for one run.

Order matters: XTTS takes pacing from the **first** clip, so name the most naturally paced one `01_*.wav`. `--speed` (from `voice.json`) is calibrated to feel the same as Kokoro; XTTS's own speed scale is slower, so `speak.py` scales it by `XTTS_SPEED_SCALE`.

Register another voice with Voicekit (the profile folder is shared by both backends):

```bash
voicekit add myname /path/to/clean_sample.wav --i-have-permission
python baymax_live.py --clone-voice myname
```

Add the XTTS clips **after** registering the voice: `voicekit add --overwrite` recreates the profile folder and removes `xtts_refs/`. Without them XTTS falls back to `sample.wav`, which still works but clones from a single clip.

**Voicekit / SV2TTS.** `--voice-backend voicekit` uses the older Voicekit cloner (the `voice-cloning` package). It is available for comparison, and `voice_clone.py` keeps its three models resident so a sentence no longer reloads them; XTTS is faster and much closer to the reference voice.

**Kokoro presets.** `--voice-backend kokoro` uses Kokoro's built-in voices. `voice_lab.py` renders 8 voices (including two blends) in 3 characters each: `calm`, `deeper` and `big_soft`. Lower pitch makes a voice feel bigger, softness smooths harsh high frequencies, and blends mix two voices into a new one.

Cloning the recorded voice of a real person, or of a copyrighted character, needs permission from the voice's owner. Use it for personal experiments only, and check the law where you live.

## Safety design

| Layer | What it catches | What it cannot catch |
|---|---|---|
| **Safety gate** | Crisis and self-harm wording; cardiac, breathing, stroke, bleeding, unconscious, seizure, poisoning and allergy emergencies; "do I have cancer/TB/diabetes" style questions | Unusual phrasing, misspellings, other languages. It leans towards triggering when unsure. |
| **Output guard** | Numbers, medicine and test names, schedules, promises, markdown, emojis, replies longer than 5 sentences | A wrong statement that contains none of those |
| **Remedy list** | Keeps remedy facts in one reviewable file; unknown topics get a fixed "no reliable remedy" reply | Mistakes in the entries themselves (not clinician-reviewed) |
| **Model training** | Style, tone, refusing doses, redirecting off-topic questions | Medical knowledge. Facts come from the base model. |

The emergency number and the crisis helpline are settings at the top of `safety_gate.py` (`EMERGENCY_NUMBER`, `CRISIS_HELPLINE`). **Check them for your country before relying on them.** Both are India's numbers (112 and Tele-MANAS 14416) and are not verified by this project.

## Tests

```bash
python test_safety.py       # 79 checks: gate triggers and non-triggers, output guard behaviour
python test_activation.py   # 40 checks: wake phrases, health routing, check-in, remedies, what may go online
python test_memory.py       # 9 checks: name, allergies, pain log, forgetting, privacy
python test_speech.py       # sentence buffer, speech queue (incl. timeout), voice shaping, guard in the stream
python test_audio_cache.py  # 8 checks: pre-rendered audio and common openers
python test_fast_first.py   # 7 checks: the optional fast first clause
python test_server.py       # 15 checks of the web server with fake models (pip install fastapi uvicorn httpx)
python test_backends.py     # 12 checks of the online provider chain against a fake server
python latency.py           # self-test of the timing code
python remedies.py          # validates every remedy entry
```

These need no GPU or model files (the voice tests also run real synthesis when the Kokoro model is present). `test_adapter.py`, `test_friend.py` and `test_checkin.py` need a trained adapter. When you find a message the gate should have caught (or wrongly caught), add it to `test_safety.py` first, then fix the rules until everything passes.

## Known limitations

- The model was trained on roughly 1,300 distinct examples, mostly synthetic. It learned a **style**, not medicine. In early tests it gave wrong facts (for example incorrect insulin storage advice) and falsely reassuring answers to serious-disease questions. The gate, the guard and the remedy list exist because of this.
- Even after training, the 1.5B model sometimes ignores "say you have no remedy" on topics it has not seen. That case is therefore handled in code, not by the model.
- The gate is keyword-based and will miss things. Treat it as a safety net, not a guarantee.
- The remedy list is general first-aid from general knowledge and has not been reviewed by a clinician.
- Friend mode relies on hand-written examples plus a system prompt; a small model can still ramble or repeat itself in long chats.
- Timing numbers come from one laptop and synthetic test speech; real speech and other hardware will differ.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Can't find 'adapter_config.json'` plus a Hugging Face 401 error | The adapter folder does not exist (training not finished, or saved elsewhere). `ls outputs` to check. A checkpoint folder works as an adapter too. |
| Out of memory while training | Set `per_device_train_batch_size=1` with `gradient_accumulation_steps=16`, lower `MAX_LEN` to 384, and free the GPU with `ollama stop <model>`. |
| `libcublas.so.12 is not found` | See "Whisper on the GPU" above. |
| `Voice runs on: CPUExecutionProvider` | The GPU libraries for the voice did not load. Check `pip list \| grep -i -E "onnxruntime\|nvidia"`; only `onnxruntime-gpu` should be installed, not `onnxruntime`. |
| A voice sentence takes seconds | Run `python prerender.py`, and check the voice really runs on the GPU. |
| Pasted a file and got a `SyntaxError`, or a file is twice its size | A stray quote or a duplicated block from pasting. Copy the downloaded file with `cp` instead of pasting, then `python -m py_compile <file>`. |
| Nothing is heard by `listen.py` | `python listen.py --list-devices`, then `--mic N`. |

## Roadmap

- Raspberry Pi 3 as the microphone and speaker, with the laptop as the brain (WebSocket, token, rate limits; Tailscale for remote use).
- Have a clinician or trusted first-aid source review `remedies.py`.
- Tune the 600 ms end-of-speech wait against real speech.
- Possibly a web client and push-to-talk.

## Data and licences

The code is released under the MIT licence (see `LICENSE`). The training data comes from public medical Q&A sets (MedQuAD and ChatDoctor-HealthCareMagic) and was rewritten by a local language model. Check the terms of those datasets, and of the base model, before redistributing any data or trained weights; this repository does not include them.

## Credits

[Qwen2.5](https://github.com/QwenLM/Qwen2.5), [PEFT](https://github.com/huggingface/peft) and [TRL](https://github.com/huggingface/trl), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggml-org/llama.cpp), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Silero VAD](https://github.com/snakers4/silero-vad), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), [ONNX Runtime](https://onnxruntime.ai), and the Groq and OpenRouter APIs. See each project for its licence.
