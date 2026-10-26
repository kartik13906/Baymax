"""
test_backends.py - tests the Groq -> OpenRouter fallback chain against a FAKE local server.
No internet and no real API keys are used.

Run:   python test_backends.py
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from llm_backends import FriendChain, NoBackendAvailable, build_chain

results = []


def check(name):
    def wrap(fn):
        try:
            fn()
            results.append((True, name, ""))
        except Exception as e:
            results.append((False, name, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


# ---------------------------------------------------------------
# Fake server. Each path prefix (/a, /b) has its own behaviour and a request counter.
# ---------------------------------------------------------------
BEHAVIOUR = {}
HITS = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        prefix = self.path.split("/")[1]
        HITS[prefix] = HITS.get(prefix, 0) + 1
        mode = BEHAVIOUR.get(prefix, "ok")
        if mode in ("429", "401", "500"):
            self.send_response(int(mode))
            if mode == "429":
                self.send_header("retry-after", "120")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        pieces = ["Hey ", "there! ", "How are ", "you?"]
        if mode == "cut":
            pieces = pieces[:2]
        if mode == "think":
            pieces = ["<thi", "nk>I should be warm.</thi", "nk>Hey ", "there!"]
        if mode == "onlythink":
            pieces = ["<think>endless pondering"]
        self.wfile.write(b": keep-alive comment\n\n")
        for p in pieces:
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) + "\n\n").encode())
            self.wfile.flush()
        if mode == "cut":
            self.wfile.write(("data: " + json.dumps({"error": {"message": "boom"}}) + "\n\n").encode())
            return
        self.wfile.write(b"data: [DONE]\n\n")


server = HTTPServer(("127.0.0.1", 0), Handler)
PORT = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def make(clock=None, keys=("a", "b")):
    clock = clock or Clock()
    os.environ["TEST_KEY_A"], os.environ["TEST_KEY_B"] = ("k" if "a" in keys else ""), ("k" if "b" in keys else "")
    chain = build_chain(order=["groq", "openrouter"], log=lambda *_: None, groq={
        "base_url": f"http://127.0.0.1:{PORT}/a", "key_env": "TEST_KEY_A", "clock": clock, "timeout": 3},
        openrouter={"base_url": f"http://127.0.0.1:{PORT}/b", "key_env": "TEST_KEY_B", "clock": clock, "timeout": 3})
    return chain, clock


def reset(a="ok", b="ok"):
    BEHAVIOUR.update({"a": a, "b": b})
    HITS.clear()


MSG = [{"role": "user", "content": "hi"}]


@check("first provider answers and its text streams in order")
def _():
    reset()
    chain, _ = make()
    assert "".join(chain.stream(MSG)) == "Hey there! How are you?"
    assert chain.last_used == "groq" and HITS == {"a": 1}


@check("rate limit on the first provider switches to the second")
def _():
    reset(a="429")
    chain, _ = make()
    assert "".join(chain.stream(MSG)) == "Hey there! How are you?"
    assert chain.last_used == "openrouter" and HITS == {"a": 1, "b": 1}


@check("a cooling-down provider is skipped without another request, then retried after the cool-down")
def _():
    reset(a="429")
    chain, clock = make()
    list(chain.stream(MSG))
    list(chain.stream(MSG))
    assert HITS == {"a": 1, "b": 2}, HITS                  # the second call did not touch provider a
    BEHAVIOUR["a"] = "ok"
    clock.now += 121                                        # retry-after said 120 s
    assert "".join(chain.stream(MSG)) and chain.last_used == "groq"


@check("repeated rate limits wait longer each time")
def _():
    reset(a="429", b="429")
    chain, clock = make()
    p = chain.providers[0]
    p.block(None)
    first = p.blocked_until - clock.now
    p.block(None)
    second = p.blocked_until - clock.now
    assert second > first, (first, second)


@check("a bad key disables that provider for the session")
def _():
    reset(a="401")
    chain, clock = make()
    list(chain.stream(MSG))
    BEHAVIOUR["a"] = "ok"
    clock.now += 100000
    list(chain.stream(MSG))
    assert chain.last_used == "openrouter" and HITS.get("a") == 1


@check("a server error cools the provider down briefly and uses the next one")
def _():
    reset(a="500")
    chain, _ = make()
    assert "".join(chain.stream(MSG)) and chain.last_used == "openrouter"


@check("a provider without a key is skipped silently")
def _():
    reset()
    chain, _ = make(keys=("b",))
    assert "".join(chain.stream(MSG)) and chain.last_used == "openrouter" and "a" not in HITS


@check("when every provider fails, NoBackendAvailable is raised before any text")
def _():
    reset(a="429", b="500")
    chain, _ = make()
    got = []
    try:
        for piece in chain.stream(MSG):
            got.append(piece)
    except NoBackendAvailable:
        pass
    else:
        raise AssertionError("did not raise")
    assert got == []


@check("a stream cut in the middle ends quietly with the text so far (no switch mid-sentence)")
def _():
    reset(a="cut")
    chain, _ = make()
    assert "".join(chain.stream(MSG)) == "Hey there! " and chain.last_used == "groq" and "b" not in HITS


@check("hidden <think> reasoning is never passed on, even when the tag is split across pieces")
def _():
    reset(a="think")
    chain, _ = make()
    assert "".join(chain.stream(MSG)) == "Hey there!" and chain.last_used == "groq"


@check("an answer that is only thinking counts as empty and moves on to the next provider")
def _():
    reset(a="onlythink")
    chain, _ = make()
    assert "".join(chain.stream(MSG)) == "Hey there! How are you?" and chain.last_used == "openrouter"


@check("no keys at all: raises NoBackendAvailable and sends nothing")
def _():
    reset()
    chain, _ = make(keys=())
    try:
        list(chain.stream(MSG))
    except NoBackendAvailable:
        assert HITS == {}
    else:
        raise AssertionError("did not raise")


failed = [r for r in results if not r[0]]
for ok, name, err in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n         {err}"))
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
