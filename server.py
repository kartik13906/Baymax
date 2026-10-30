"""
server.py - Baymax as a small web server (the Docker image runs this).

    browser (or a Raspberry Pi client)                       this computer / container
    ---------------------------------                       ---------------------------------------------
    microphone -> 16 kHz PCM  --WebSocket-->  Silero VAD -> Whisper -> Session (safety gate, remedies,
                                                                       memory, local model, friend chat)
    speaker    <- WAV sentences <--WebSocket--  Kokoro voice (pre-rendered lines play instantly)

One WebSocket = one Session (its own wake state, pain check-in and chat history). The heavy models
(Whisper, Kokoro) are loaded once and shared; a lock keeps two people from using one model at the same time.

Messages
  client -> server   binary: int16 little-endian PCM, 16 kHz, mono, any length
                     text:   {"type": "text", "text": "..."}          (typed chat)
  server -> client   binary: one WAV file per spoken sentence, in order
                     text:   {"type": "user",  "text": ...}           what Baymax heard
                             {"type": "baymax", "text": ...}         a sentence Baymax says
                             {"type": "state", "state": "sleeping"|"active", "result": ...}
                             {"type": "turn_end"}
                             {"type": "error", "text": ...}

Safety of the server itself
  * Listens on 127.0.0.1 unless you say otherwise. Anyone who can open the page can talk to Baymax.
  * BAYMAX_TOKEN (optional) = a shared password; the page asks for it.
  * Browsers from another website are refused (Origin must match the Host).
  * At most --max-sessions people at once, 500 characters per typed message, 30 turns a minute.
  * Health talk, memory and safety replies never leave the server. Only casual chat can go to Groq/OpenRouter,
    and only if you put API keys in .env.
"""

import argparse
import asyncio
import io
import json
import os
import queue
import secrets
import threading
import time
import wave
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

SAMPLE_RATE = 16000
CHUNK = 512                     # Silero VAD takes exactly 512 samples at 16 kHz
MIN_SPEECH_SEC = 0.3
MAX_TEXT_CHARS = 500
MAX_FRAME_BYTES = 64_000        # one browser audio frame is a few KB; anything bigger is not from our page
TURNS_PER_MINUTE = 30
WEB_DIR = Path(__file__).resolve().parent / "web"

ERROR_MODEL = "I cannot reach my health model right now. Please try again in a moment."
ERROR_SLOW_DOWN = "Please slow down a little. I can only answer so many messages a minute."


def wav_bytes(audio, sample_rate):
    """float32 mono audio -> a small 16-bit WAV file in memory (what the browser decodes and plays)."""
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------
# 1. ONE CONNECTION = ONE SESSION, run in its own thread
#    (a turn blocks while the model writes and the voice speaks, so it must not run in the web loop)
# ---------------------------------------------------------------
class Conn:
    def __init__(self, engine, session_factory, send_json, send_bytes, log=print):
        self.engine, self.session_factory = engine, session_factory
        self.send_json, self.send_bytes, self.log = send_json, send_bytes, log
        self.inbox = queue.Queue()
        self.turns = deque()                                   # times of recent turns, for the rate limit
        self.session = None
        self._close_session = lambda: None
        self._leftover = np.zeros(0, dtype=np.float32)
        self._thread = threading.Thread(target=self._run, daemon=True)

    # called from the web loop ------------------------------------------------
    def start(self):
        self._thread.start()

    def push_audio(self, pcm: bytes):
        self.inbox.put(("audio", pcm))

    def push_text(self, text: str):
        self.inbox.put(("text", text))

    def stop(self):
        self.inbox.put(("stop", None))

    # used by the session factory ----------------------------------------------
    def send_wav(self, audio, sample_rate):
        self.send_bytes(wav_bytes(audio, sample_rate))

    def show(self, sentence):
        self.send_json({"type": "baymax", "text": sentence})

    # the thread -----------------------------------------------------------------
    def _run(self):
        from listen import UtteranceBuffer
        try:
            self.session, self._close_session = self.session_factory(self)
            self.vad = self.engine.new_vad()
            self.buf = UtteranceBuffer()
            self.send_json({"type": "state", "state": self.session.state, "result": "ready"})
        except Exception as e:                                 # model missing, etc.: say so, do not hang
            self.log(f"[server] could not start a session: {e!r}")
            self.send_json({"type": "error", "text": ERROR_MODEL})
            return
        try:
            while True:
                kind, payload = self.inbox.get()
                if kind == "stop":
                    break
                if kind == "text":
                    self._turn(payload)
                    self._forget_audio()
                elif kind == "audio":
                    self._feed(payload)
        finally:
            try:
                self._close_session()
            except Exception:
                pass

    def _feed(self, pcm):
        samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        data = np.concatenate([self._leftover, samples])
        n = len(data) // CHUNK
        self._leftover = data[n * CHUNK:]
        for i in range(n):
            chunk = data[i * CHUNK:(i + 1) * CHUNK]
            utterance = self.buf.feed(chunk, self.vad(chunk))
            if utterance is None:
                continue
            self.vad.reset()
            if len(utterance) / SAMPLE_RATE < MIN_SPEECH_SEC:
                continue
            text = self.engine.transcribe(utterance)
            if text:
                self._turn(text)
                self._forget_audio()                           # Baymax just spoke: drop what the mic heard meanwhile
            return

    def _forget_audio(self):
        keep = []
        while True:
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                break
            if item[0] != "audio":
                keep.append(item)
        for item in keep:
            self.inbox.put(item)
        self.buf.reset()
        self.vad.reset()
        self._leftover = np.zeros(0, dtype=np.float32)

    def _allowed_now(self):
        now = time.monotonic()
        while self.turns and now - self.turns[0] > 60:
            self.turns.popleft()
        if len(self.turns) >= TURNS_PER_MINUTE:
            return False
        self.turns.append(now)
        return True

    def _turn(self, text):
        text = (text or "").strip()[:MAX_TEXT_CHARS]
        if not text:
            return
        if not self._allowed_now():
            self.send_json({"type": "error", "text": ERROR_SLOW_DOWN})
            return
        self.send_json({"type": "user", "text": text})
        try:
            self.session.marks.clear()
            result = self.session.handle(text)
        except Exception as e:                                 # e.g. Ollama not reachable: never leave the page hanging
            self.log(f"[server] turn failed: {e!r}")
            self.send_json({"type": "error", "text": ERROR_MODEL})
            result = "error"
        self.send_json({"type": "state", "state": self.session.state, "result": result})
        self.send_json({"type": "turn_end"})


# ---------------------------------------------------------------
# 2. THE WEB APP (kept free of model code so it can be tested with fakes)
# ---------------------------------------------------------------
def create_app(engine, session_factory, token="", max_sessions=2, allowed_origins=(), log=print):
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title="Baymax", docs_url=None, redoc_url=None, openapi_url=None)
    active = {"n": 0}
    lock = threading.Lock()
    allowed = {o.strip().lower() for o in allowed_origins if o.strip()}

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.get("/healthz")
    async def healthz():
        return JSONResponse({"ok": True, "sessions": active["n"], "token_required": bool(token)})

    def origin_ok(ws):
        origin = ws.headers.get("origin")
        if not origin:                                         # not a browser (e.g. the Pi client): the token protects it
            return True
        netloc = urlparse(origin).netloc.lower()
        return netloc == (ws.headers.get("host") or "").lower() or origin.lower() in allowed or netloc in allowed

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        if not origin_ok(ws):
            await ws.close(code=4403)
            return
        await ws.accept()
        supplied = ws.query_params.get("token", "")
        if token and not secrets.compare_digest(supplied.encode(), token.encode()):
            await ws.send_text(json.dumps({"type": "auth_required"}))
            await ws.close(code=4401)
            return
        with lock:
            full = active["n"] >= max_sessions
            if not full:
                active["n"] += 1
        if full:
            await ws.send_text(json.dumps({"type": "error", "text": "Baymax is busy with other people right now. Please try again soon."}))
            await ws.close(code=4503)
            return

        loop = asyncio.get_running_loop()
        out = asyncio.Queue()
        send_json = lambda d: loop.call_soon_threadsafe(out.put_nowait, json.dumps(d))
        send_bytes = lambda b: loop.call_soon_threadsafe(out.put_nowait, b)
        conn = Conn(engine, session_factory, send_json, send_bytes, log=log)

        async def sender():
            while True:
                message = await out.get()
                if message is None:
                    return
                if isinstance(message, bytes):
                    await ws.send_bytes(message)
                else:
                    await ws.send_text(message)

        conn.start()
        task = asyncio.create_task(sender())
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    if len(data) <= MAX_FRAME_BYTES:
                        conn.push_audio(data)
                    continue
                text = message.get("text")
                if text:
                    try:
                        payload = json.loads(text)
                    except ValueError:
                        continue
                    if isinstance(payload, dict) and payload.get("type") == "text" and isinstance(payload.get("text"), str):
                        conn.push_text(payload["text"][:MAX_TEXT_CHARS])
        except Exception as e:
            log(f"[server] connection ended: {e!r}")
        finally:
            conn.stop()
            with lock:
                active["n"] -= 1
            out.put_nowait(None)
            try:
                await asyncio.wait_for(task, 2)
            except Exception:
                task.cancel()

    return app


# ---------------------------------------------------------------
# 3. THE REAL MODELS (loaded once, shared by every connection)
# ---------------------------------------------------------------
class _Vad:
    """Silero VAD for one connection (it keeps state, so connections cannot share one)."""
    def __init__(self, threshold):
        import torch
        from listen import load_vad
        self._torch, self._vad = torch, load_vad(threshold)

    def __call__(self, chunk):
        return self._vad(self._torch.from_numpy(chunk))

    def reset(self):
        self._vad.reset_states()


class Engine:
    def __init__(self, args):
        from listen import load_stt
        from speak import load_tts
        self.args = args
        self.stt, self.stt_device = load_stt(args.stt_device)
        print("Loading the voice ...")
        synth, self.sample_rate, _ = load_tts(args.model_dir, args.voice, args.speed, args.pitch, args.softness)
        if not args.no_cache:
            from audio_cache import AudioCache, voice_key
            cache = AudioCache(args.audio_cache, voice_key(args.voice, args.speed, args.pitch, args.softness))
            ready = cache.load()
            if ready:
                synth = cache.wrap(synth)
                print(f"Pre-rendered audio: {ready} fixed lines play instantly.")
            else:
                print("No pre-rendered audio found (python prerender.py --voice-backend kokoro). Fixed lines will be a little slower.")
        self._synth = synth
        self._tts_lock, self._stt_lock = threading.Lock(), threading.Lock()

    def transcribe(self, audio):
        from listen import transcribe
        with self._stt_lock:
            return transcribe(self.stt, audio)

    def synth(self, text):
        with self._tts_lock:
            return self._synth(text)

    def new_vad(self):
        return _Vad(self.args.threshold)


def real_session_factory(engine, args):
    """Builds a Session wired to this connection: its voice sends WAV sentences to the browser."""
    def factory(conn):
        from baymax_live import make_session
        from speak import SpeechQueue
        speech = SpeechQueue(engine.synth, play_fn=conn.send_wav)
        speaker = (speech.say, speech.wait, speech.close, speech.start_timer, speech.ttfa_ms)
        session, close = make_session(args, speaker=speaker, show=conn.show)
        return session, close
    return factory


def main():
    from speak import DEFAULT_MODEL_DIR, load_voice_settings
    settings = load_voice_settings()
    env = os.environ.get
    ap = argparse.ArgumentParser(description="Baymax web server")
    ap.add_argument("--host", default=env("BAYMAX_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("BAYMAX_PORT", "8000")))
    ap.add_argument("--max-sessions", type=int, default=int(env("BAYMAX_MAX_SESSIONS", "2")))
    ap.add_argument("--stt-device", default=env("BAYMAX_STT_DEVICE", "cpu"), choices=["cuda", "cpu"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--memory", default=env("BAYMAX_MEMORY", "baymax_memory.db"))
    ap.add_argument("--no-memory", action="store_true", default=env("BAYMAX_MEMORY", "") == "off")
    ap.add_argument("--local-friend", action="store_true", default=env("BAYMAX_LOCAL_FRIEND", "") == "1")
    ap.add_argument("--fast-first", action="store_true", default=env("BAYMAX_FAST_FIRST", "") == "1")
    ap.add_argument("--audio-cache", default=env("BAYMAX_AUDIO_CACHE", "audio_cache"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--voice", default=settings["voice"])
    ap.add_argument("--speed", type=float, default=settings["speed"])
    ap.add_argument("--pitch", type=float, default=settings["pitch"])
    ap.add_argument("--softness", type=float, default=settings["softness"])
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--ssl-certfile", default=env("BAYMAX_SSL_CERT", ""))
    ap.add_argument("--ssl-keyfile", default=env("BAYMAX_SSL_KEY", ""))
    args = ap.parse_args()
    args.online_friend, args.no_audio, args.silent = False, False, False
    args.voice_backend, args.clone_voice = "kokoro", "baymax"     # the server only uses the preset voice: no personal voices are ever shipped

    from llm_backends import load_env_file
    load_env_file()
    if not args.local_friend and (os.environ.get("GROQ_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        args.online_friend = True

    token = env("BAYMAX_TOKEN", "")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        print("WARNING: Baymax is reachable from other devices and BAYMAX_TOKEN is not set: anyone who can reach this")
        print("         address can talk to it (and see what it remembers). Set BAYMAX_TOKEN in .env to require a password.\n")

    engine = Engine(args)
    app = create_app(engine, real_session_factory(engine, args), token=token, max_sessions=args.max_sessions,
                     allowed_origins=env("BAYMAX_ALLOWED_ORIGINS", "").split(","))
    from baymax_speak import warmup_model
    warmup_model()
    import uvicorn
    scheme = "https" if args.ssl_certfile else "http"
    print(f"\nBaymax is ready: {scheme}://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                ssl_certfile=args.ssl_certfile or None, ssl_keyfile=args.ssl_keyfile or None)


if __name__ == "__main__":
    main()
