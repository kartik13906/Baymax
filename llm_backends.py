"""
llm_backends.py - online chat providers for Baymax's FRIEND mode only.

    Groq  --(rate limit / error)-->  OpenRouter  --(rate limit / error)-->  (caller falls back to the local model)

Both services speak the same "OpenAI-style" protocol, so one small class handles both.
Only the Python standard library is used (no extra install).

Rules this file follows:
  * It is only ever called for casual chat. The health / care path never touches it.
  * API keys come from environment variables (or a .env file that you DO NOT commit).
  * A provider that hits its limit is put on a cool-down and skipped, so Baymax does not
    waste time retrying it on every message. It is tried again when the cool-down ends.
  * If a provider fails after it has already started answering, the sentence stream just ends
    (we cannot swap voices in the middle of a reply).

    chain = build_chain()
    for piece in chain.stream(messages):       # messages = [{"role": ..., "content": ...}, ...]
        ...
    chain.last_used                            # "groq", "openrouter" ...
Raises NoBackendAvailable (before any text) when nobody can answer: the caller uses the local model.

Check your setup:   python llm_backends.py --check
"""

import json
import os
import socket
import time
import urllib.error
import urllib.request

# Model names change and free models get withdrawn. Override them with the environment
# variables below, and verify the names on the provider's website.
PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}
PROVIDER_ORDER = ["groq", "openrouter"]

MAX_COOLDOWN = 3600        # never wait more than an hour before trying a provider again
DEFAULT_RETRY_AFTER = 60   # seconds, when the provider does not say how long to wait
ERROR_COOLDOWN = 30        # seconds after a timeout or a server error


class NoBackendAvailable(Exception):
    """Every provider is missing a key, cooling down, or failed."""


class RateLimited(Exception):
    def __init__(self, retry_after=None, detail=""):
        super().__init__(f"rate limited (retry after {retry_after})")
        self.retry_after = retry_after
        self.detail = detail


class Unusable(Exception):
    """Bad key, or the model no longer exists: do not try this provider again this session."""


class TemporaryFailure(Exception):
    """Timeout, network problem or server error: try again later."""


DEFAULT_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env_file(path=None):
    """Tiny .env reader: KEY=VALUE lines. Does not override variables that are already set.
    By default it reads the .env next to this file, so starting Baymax from another
    folder still finds the keys."""
    path = path or DEFAULT_ENV_FILE
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


class ThinkFilter:
    """Some models write their reasoning inside <think> ... </think> before the answer. That must never
    be spoken. Works on a stream, even when a tag is split across two pieces ("<thi" + "nk>")."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, text):
        self.buf += text
        out = ""
        while True:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    self.buf = self.buf[-(len(self.CLOSE) - 1):]           # keep a possible half-written tag
                    return out
                self.buf = self.buf[i + len(self.CLOSE):]
                self.inside = False
            else:
                i = self.buf.find(self.OPEN)
                if i == -1:
                    keep = 0
                    for k in range(min(len(self.OPEN) - 1, len(self.buf)), 0, -1):
                        if self.OPEN.startswith(self.buf[-k:]):
                            keep = k
                            break
                    out += self.buf[:len(self.buf) - keep]
                    self.buf = self.buf[len(self.buf) - keep:]
                    return out
                out += self.buf[:i]
                self.buf = self.buf[i + len(self.OPEN):]
                self.inside = True

    def flush(self):
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


class Provider:
    def __init__(self, name, base_url, key_env, model_env, model, clock=time.monotonic, timeout=8.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.key_env = key_env
        self.model = os.environ.get(model_env) or model
        self.clock = clock
        self.timeout = timeout
        self.blocked_until = 0.0
        self.fail_count = 0
        self.disabled = False

    # -- availability -------------------------------------------
    @property
    def key(self):
        return os.environ.get(self.key_env, "").strip()

    def available(self):
        return bool(self.key) and not self.disabled and self.clock() >= self.blocked_until

    def status(self):
        if not self.key:
            return f"no key ({self.key_env} is not set)"
        if self.disabled:
            return "disabled (bad key or model)"
        wait = self.blocked_until - self.clock()
        return f"cooling down for {wait:.0f} s" if wait > 0 else "ready"

    def block(self, retry_after=None):
        """Cool down. Repeated failures wait longer: 60 s, 120 s, 240 s ... up to an hour."""
        self.fail_count += 1
        backoff = min(DEFAULT_RETRY_AFTER * 2 ** (self.fail_count - 1), MAX_COOLDOWN)
        wait = min(max(retry_after or 0, backoff), MAX_COOLDOWN)
        self.blocked_until = self.clock() + wait
        return wait

    @staticmethod
    def _error_detail(e):
        try:
            raw = e.read().decode("utf-8", "replace")
            try:
                err = json.loads(raw).get("error", raw)
                raw = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            except (ValueError, AttributeError):
                pass
            return raw.strip().replace("\n", " ")[:200] or "(no message)"
        except Exception:
            return "(no message)"

    def list_models(self):
        """Ask the provider which models this key can use. Returns (ids, free_ids_or_None)."""
        req = urllib.request.Request(self.base_url + "/models", headers={
            "Authorization": f"Bearer {self.key}", "User-Agent": "baymax-prototype"})
        with urllib.request.urlopen(req, timeout=self.timeout + 7) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data", [])
        ids = sorted(m["id"] for m in items if "id" in m)
        free = None
        if "openrouter" in self.base_url and any("pricing" in m for m in items):   # only OpenRouter sells by price
            free = sorted(m["id"] for m in items
                          if str((m.get("pricing") or {}).get("prompt", "1")) in ("0", "0.0", "0.00000000"))
        return ids, free

    # -- the request --------------------------------------------
    def stream(self, messages, temperature=0.7, max_tokens=500):
        """Yield text pieces. Raises RateLimited / Unusable / TemporaryFailure before the first piece."""
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": True,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "Accept": "text/event-stream", "User-Agent": "baymax-prototype"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            code = e.code
            detail = self._error_detail(e)
            if code == 429:
                retry = e.headers.get("retry-after") if e.headers else None
                try:
                    retry = float(retry) if retry is not None else None
                except ValueError:
                    retry = None
                raise RateLimited(retry, detail)
            if code in (401, 403, 404):
                hint = {401: "the key was rejected", 403: "access refused",
                        404: f"model '{self.model}' not found or not available to you"}[code]
                raise Unusable(f"HTTP {code}: {hint}. Provider says: {detail}")
            raise TemporaryFailure(f"HTTP {code} {detail}")
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            raise TemporaryFailure(str(e))

        think = ThinkFilter()
        with resp:
            try:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue                              # keep-alive comments, blank lines
                    data = line[5:].strip()
                    if data == "[DONE]":
                        tail = think.flush()
                        if tail:
                            yield tail
                        return
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if "error" in chunk:                      # OpenRouter reports mid-stream errors this way
                        raise TemporaryFailure(str(chunk["error"]))
                    for choice in chunk.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            piece = think.feed(piece)
                            if piece:
                                yield piece
            except (socket.timeout, TimeoutError, ConnectionError, OSError) as e:
                raise TemporaryFailure(str(e))


class FriendChain:
    def __init__(self, providers, log=print):
        self.providers = providers
        self.log = log
        self.last_used = None

    def stream(self, messages, temperature=0.7, max_tokens=500):
        self.last_used = None
        for p in self.providers:
            if not p.available():
                continue
            gen = p.stream(messages, temperature, max_tokens)
            try:
                first = next(gen)
            except StopIteration:
                continue                                      # empty answer: try the next one
            except RateLimited as e:
                wait = p.block(e.retry_after)
                why = f" Provider says: {e.detail}" if e.detail else ""
                self.log(f"   [{p.name}: rate limit reached, switching. Will retry in {wait:.0f} s.{why}]")
                continue
            except Unusable as e:
                p.disabled = True
                self.log(f"   [{p.name}: {e}. Check the key and model name; skipping it this session]")
                continue
            except TemporaryFailure as e:
                p.block(ERROR_COOLDOWN)
                self.log(f"   [{p.name}: problem ({e}), switching]")
                continue
            p.fail_count = 0
            self.last_used = p.name
            yield first
            try:
                yield from gen
            except (RateLimited, Unusable, TemporaryFailure) as e:
                self.log(f"   [{p.name}: cut off mid-answer ({e}); keeping what was said]")
            return
        raise NoBackendAvailable("no online provider could answer")

    def describe(self):
        return ", ".join(f"{p.name} ({p.model}): {p.status()}" for p in self.providers)


def build_chain(order=None, log=print, **overrides):
    """overrides: per-provider dict for tests, e.g. {"groq": {"base_url": "http://127.0.0.1:9000/a"}}."""
    providers = []
    for name in order or PROVIDER_ORDER:
        cfg = dict(PROVIDER_DEFAULTS[name])
        cfg.update(overrides.get(name, {}))
        providers.append(Provider(name, cfg["base_url"], cfg["key_env"], cfg["model_env"], cfg["model"],
                                  **{k: v for k, v in cfg.items() if k in ("clock", "timeout")}))
    return FriendChain(providers, log=log)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="send 'hi' to each provider and report")
    ap.add_argument("--models", action="store_true", help="list the model names each provider offers you")
    args = ap.parse_args()
    load_env_file()
    chain = build_chain()
    print("Providers:", chain.describe(), "\n")
    if args.models:
        for p in chain.providers:
            print(f"=== {p.name} ===")
            if not p.key:
                print(f"  no key ({p.key_env} is not set)\n")
                continue
            try:
                ids, free = p.list_models()
            except Exception as e:
                print(f"  could not list models: {e}\n")
                continue
            shown = free if free else ids                      # an empty "free" list means: show everything
            label = "free models" if free else "models (Groq: pick a chat model; ignore whisper, tts and guard ones)"
            print(f"  {len(shown)} {label}:")
            for m in shown:
                print("   ", m)
            print()
        print("Put the one you want in .env, for example:  GROQ_MODEL=<name>   OPENROUTER_MODEL=<name>")
        return
    for p in chain.providers:
        if not p.available():
            print(f"{p.name}: skipped ({p.status()})")
            continue
        single = FriendChain([p])
        try:
            t = time.perf_counter()
            text = "".join(single.stream([{"role": "user", "content": "Say hi in five words."}], max_tokens=300))
            print(f"{p.name}: OK in {time.perf_counter() - t:.1f} s -> {text.strip()!r}")
        except NoBackendAvailable:
            print(f"{p.name}: FAILED (see the message above)")


if __name__ == "__main__":
    main()
