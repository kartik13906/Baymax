"""
test_server.py - tests for server.py with fake models (no Whisper, Kokoro or Ollama needed).

Run:  python test_server.py
Needs:  pip install fastapi uvicorn httpx
"""

import json
import sys
import time

import numpy as np

from server import (CHUNK, ERROR_MODEL, MAX_TEXT_CHARS, TURNS_PER_MINUTE, create_app, wav_bytes)

results = []


def check(name):
    def deco(fn):
        try:
            fn()
            results.append((True, name, ""))
        except Exception as e:                                  # noqa: BLE001
            import traceback
            results.append((False, name, "".join(traceback.format_exception_only(type(e), e)).strip() + "\n" + traceback.format_exc(limit=3)))
        return fn
    return deco


# ---------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------
class FakeVad:
    """Loud chunk = speech starts; two quiet chunks after speech = it ended."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.speaking, self.quiet = False, 0

    def __call__(self, chunk):
        loud = float(np.abs(chunk).mean()) > 0.05
        if loud and not self.speaking:
            self.speaking, self.quiet = True, 0
            return {"start": 0}
        if self.speaking and not loud:
            self.quiet += 1
            if self.quiet >= 2:
                self.speaking = False
                return {"end": 0}
        return None


class FakeEngine:
    def __init__(self, text="hello baymax"):
        self.text, self.heard = text, []

    def new_vad(self):
        return FakeVad()

    def transcribe(self, audio):
        self.heard.append(len(audio))
        return self.text


class FakeSession:
    def __init__(self, conn, fail=False):
        self.conn, self.fail = conn, fail
        self.state, self.marks, self.said = "sleeping", {}, []

    def handle(self, text):
        if self.fail:
            raise RuntimeError("ollama is down")
        self.said.append(text)
        self.state = "active"
        self.conn.show("I am here.")
        self.conn.send_wav(np.zeros(240, dtype=np.float32), 24000)
        return "woke"


def make_client(token="", max_sessions=2, fail=False, engine=None, **kw):
    from fastapi.testclient import TestClient
    sessions = []

    def factory(conn):
        s = FakeSession(conn, fail=fail)
        sessions.append(s)
        return s, (lambda: None)

    app = create_app(engine or FakeEngine(), factory, token=token, max_sessions=max_sessions, log=lambda *_: None, **kw)
    return TestClient(app), sessions


def collect(ws, until="turn_end", limit=20):
    """Read messages until one of the given type. Returns (json messages, binary messages)."""
    texts, blobs = [], []
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            blobs.append(m["bytes"])
            continue
        d = json.loads(m["text"])
        texts.append(d)
        if d["type"] == until:
            break
    return texts, blobs


def loud_pcm(n_chunks):
    return (np.full(CHUNK * n_chunks, 0.3) * 32767).astype("<i2").tobytes()


def quiet_pcm(n_chunks):
    return np.zeros(CHUNK * n_chunks, dtype="<i2").tobytes()


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------
@check("the page, the health check and the security headers")
def _():
    c, _s = make_client()
    r = c.get("/")
    assert r.status_code == 200 and "Baymax" in r.text and "not a medical device" in r.text
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"
    h = c.get("/healthz").json()
    assert h["ok"] is True and h["token_required"] is False
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


@check("typed message: the browser gets what was heard, the sentence, the audio, the new state and turn_end")
def _():
    c, sessions = make_client()
    with c.websocket_connect("/ws") as ws:
        first, _ = collect(ws, until="state")
        assert first[-1]["result"] == "ready" and first[-1]["state"] == "sleeping"
        ws.send_text(json.dumps({"type": "text", "text": "hello baymax"}))
        texts, blobs = collect(ws)
    kinds = [t["type"] for t in texts]
    assert kinds[:2] == ["user", "baymax"] and kinds[-1] == "turn_end" and "state" in kinds, kinds
    assert texts[0]["text"] == "hello baymax" and texts[-2]["state"] == "active"
    assert len(blobs) == 1 and blobs[0][:4] == b"RIFF" and blobs[0][8:12] == b"WAVE"
    assert sessions[0].said == ["hello baymax"]


@check("spoken audio: voice detected, transcribed, answered; the mic echo that arrives meanwhile is dropped")
def _():
    eng = FakeEngine("ouch")
    c, sessions = make_client(engine=eng)
    with c.websocket_connect("/ws") as ws:
        collect(ws, until="state")
        # 0.3 s needs ~10 chunks of speech to count; then silence ends it
        ws.send_bytes(loud_pcm(12) + quiet_pcm(3))
        texts, blobs = collect(ws)
    assert [t["type"] for t in texts][0] == "user" and texts[0]["text"] == "ouch"
    assert sessions[0].said == ["ouch"] and len(eng.heard) == 1 and blobs


@check("a very short noise blip is ignored (never reaches Whisper)")
def _():
    eng = FakeEngine()
    c, sessions = make_client(engine=eng)
    with c.websocket_connect("/ws") as ws:
        collect(ws, until="state")
        ws.send_bytes(loud_pcm(2) + quiet_pcm(3))
        ws.send_text(json.dumps({"type": "text", "text": "hi"}))        # proves the thread is alive and in order
        collect(ws)
    assert eng.heard == [] and sessions[0].said == ["hi"]


@check("audio sent in odd-sized pieces is stitched back together")
def _():
    eng = FakeEngine("hello baymax")
    c, sessions = make_client(engine=eng)
    pcm = loud_pcm(12) + quiet_pcm(3)
    with c.websocket_connect("/ws") as ws:
        collect(ws, until="state")
        for i in range(0, len(pcm), 701):                                # 701 bytes: not a multiple of a chunk or a sample pair
            ws.send_bytes(pcm[i:i + 701])
        texts, _ = collect(ws)
    assert sessions[0].said == ["hello baymax"]


@check("long typed messages are cut to the limit")
def _():
    c, sessions = make_client()
    with c.websocket_connect("/ws") as ws:
        collect(ws, until="state")
        ws.send_text(json.dumps({"type": "text", "text": "a" * 5000}))
        collect(ws)
    assert len(sessions[0].said[0]) == MAX_TEXT_CHARS


@check("garbage and oversized frames are ignored, the connection survives")
def _():
    c, sessions = make_client()
    with c.websocket_connect("/ws") as ws:
        collect(ws, until="state")
        ws.send_text("not json")
        ws.send_text(json.dumps({"type": "text", "text": 123}))
        ws.send_text(json.dumps(["list"]))
        ws.send_bytes(b"\x00" * 200_000)
        ws.send_text(json.dumps({"type": "text", "text": "still here"}))
        collect(ws)
    assert sessions[0].said == ["still here"]


@check("a model failure becomes a spoken-style error, not a hung page")
def _():
    c, _s = make_client(fail=True)
    with c.websocket_connect("/ws") as ws:
        collect(ws, until="state")
        ws.send_text(json.dumps({"type": "text", "text": "hello"}))
        texts, _ = collect(ws)
    errors = [t for t in texts if t["type"] == "error"]
    assert errors and errors[0]["text"] == ERROR_MODEL and texts[-1]["type"] == "turn_end"


@check("password: wrong or missing is refused, correct works")
def _():
    from starlette.websockets import WebSocketDisconnect
    c, _s = make_client(token="s3cret")
    with c.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "auth_required"
        try:
            ws.receive_text()
            raise AssertionError("should have been closed")
        except WebSocketDisconnect as e:
            assert e.code == 4401
    with c.websocket_connect("/ws?token=wrong") as ws:
        assert json.loads(ws.receive_text())["type"] == "auth_required"
    with c.websocket_connect("/ws?token=s3cret") as ws:
        first, _ = collect(ws, until="state")
        assert first[0]["result"] == "ready"


@check("another website cannot talk to a local Baymax (Origin must match Host)")
def _():
    from starlette.websockets import WebSocketDisconnect
    c, _s = make_client(allowed_origins=["https://baymax.example"])
    bad = {"origin": "https://evil.example"}
    try:
        with c.websocket_connect("/ws", headers=bad):
            raise AssertionError("an outside origin must be refused")
    except WebSocketDisconnect:
        pass
    with c.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:           # same host: fine
        collect(ws, until="state")
    with c.websocket_connect("/ws", headers={"origin": "https://baymax.example"}) as ws:      # allow-listed
        collect(ws, until="state")
    with c.websocket_connect("/ws") as ws:                                                    # no Origin = not a browser (Pi client)
        collect(ws, until="state")


@check("a full server turns people away politely, and a free slot opens when someone leaves")
def _():
    from starlette.websockets import WebSocketDisconnect
    c, _s = make_client(max_sessions=1)
    with c.websocket_connect("/ws") as first:
        collect(first, until="state")
        with c.websocket_connect("/ws") as second:
            msg = json.loads(second.receive_text())
            assert msg["type"] == "error" and "busy" in msg["text"]
            try:
                second.receive_text()
            except WebSocketDisconnect as e:
                assert e.code == 4503
    deadline = time.time() + 3
    while c.get("/healthz").json()["sessions"] and time.time() < deadline:
        time.sleep(0.05)
    assert c.get("/healthz").json()["sessions"] == 0
    with c.websocket_connect("/ws") as again:
        collect(again, until="state")


@check("rate limit: too many turns a minute get a gentle error")
def _():
    c, sessions = make_client()
    with c.websocket_connect("/ws") as ws:
        collect(ws, until="state")
        for i in range(TURNS_PER_MINUTE + 3):
            ws.send_text(json.dumps({"type": "text", "text": f"hi {i}"}))
        seen_error = False
        for _ in range((TURNS_PER_MINUTE + 3) * 6):
            m = ws.receive()
            if m.get("text") and json.loads(m["text"]).get("type") == "error":
                seen_error = True
                break
    assert seen_error and len(sessions[0].said) == TURNS_PER_MINUTE


@check("two connections have separate sessions")
def _():
    c, sessions = make_client()
    with c.websocket_connect("/ws") as a, c.websocket_connect("/ws") as b:
        collect(a, until="state")
        collect(b, until="state")
        a.send_text(json.dumps({"type": "text", "text": "from a"}))
        collect(a)
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert [s.said for s in sessions].count(["from a"]) == 1


@check("wav_bytes: a valid 16-bit mono WAV, clipped not wrapped")
def _():
    import io
    import wave
    data = wav_bytes(np.array([0.0, 2.0, -2.0, 0.5], dtype=np.float32), 24000)
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 4)
        frames = np.frombuffer(w.readframes(4), dtype="<i2")
    assert frames[1] == 32767 and frames[2] == -32767 and abs(int(frames[3]) - 16383) <= 1


@check("real wiring: the real Session + SpeechQueue send the safety reply as audio and text (model and voice faked)")
def _():
    import argparse

    import baymax_speak
    from server import real_session_factory

    class Eng(FakeEngine):
        def synth(self, text):
            return np.zeros(2400, dtype=np.float32), 24000

    baymax_speak.model_tokens = lambda *a, **k: iter(["Please rest and keep it raised. "])     # the local model, faked
    args = argparse.Namespace(online_friend=False, local_friend=True, no_memory=True, memory=":memory:", idle=600,
                              fast_first=False, no_audio=False, silent=False)
    eng = Eng()
    from fastapi.testclient import TestClient
    app = create_app(eng, real_session_factory(eng, args), log=print)
    with TestClient(app).websocket_connect("/ws") as ws:
        collect(ws, until="state")
        ws.send_text(json.dumps({"type": "text", "text": "i cannot breathe"}))        # safety gate: fixed reply, model not used
        texts, blobs = collect(ws)
        spoken = " ".join(t["text"] for t in texts if t["type"] == "baymax")
        assert "112" in spoken and "call" in spoken.lower(), spoken
        assert blobs and all(b[:4] == b"RIFF" for b in blobs), len(blobs)
        ws.send_text(json.dumps({"type": "text", "text": "hello baymax"}))             # wakes with a fixed greeting
        texts, _ = collect(ws)
        assert any(t["type"] == "state" and t["state"] == "active" for t in texts)


failed = [r for r in results if not r[0]]
for ok, name, err in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n         {err}"))
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
