# -*- coding: utf-8 -*-
"""Shared helpers. The public snapshot includes only synthetic demo data."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "chatgpt_export_sample" / "conversations.json"
LOG_DIR = DATA / "logs"

SUB2API_BASE = os.environ.get("AI_JOB_BASE_URL") or os.environ.get("SUB2API_BASE", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
LLM_MODEL = os.environ.get("LLM_MODEL", "")


BOILERPLATE_MARKERS = (
    "# AGENTS.md instructions",
    "<INSTRUCTIONS>",
    "# Global Codex Policy",
    "<recommended_plugins>",
    "<environment_context>",
    "<codex_internal_context",
    "<skills_instructions>",
    "<permissions instructions>",
    "<apps_instructions>",
    "======= MEMORY_SUMMARY BEGINS =======",
    "<fastctx:begin>",
    "<fastctx:end>",
    "<oai-mem-citation>",
)


def clean_user_message(text: str) -> str:
    """Strip Codex/ChatGPT system-instruction boilerplate from a user message."""
    if any(m in text for m in BOILERPLATE_MARKERS):
        return ""
    return text

_KEY: str | None = None
_KEY_LOCK = threading.Lock()


def sub2api_key() -> str:
    global _KEY
    with _KEY_LOCK:
        if _KEY is None:
            injected = os.environ.get("AI_JOB_API_KEY") or os.environ.get("LLM_API_KEY")
            if injected:
                _KEY = injected
                return _KEY
            raise RuntimeError("Set LLM_API_KEY explicitly for an online run.")
        return _KEY


def load_corpus(path: Path | str | None = None) -> list[dict]:
    """Parse the ChatGPT-export-format corpus into conversation dicts."""
    export = json.loads((Path(path) if path else CORPUS).read_text(encoding="utf-8"))
    convs = []
    for c in export["conversations"]:
        mapping = c.get("mapping") or {}
        def _ct(m):
            v = m.get("create_time")
            try:
                return float(v)
            except (TypeError, ValueError):
                try:
                    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError):
                    return 0.0

        items = sorted(mapping.values(), key=_ct)
        messages = []
        for it in items:
            author = (it.get("author") or {}).get("role", "")
            if author not in ("user", "assistant"):
                continue
            parts = []
            for ct in it.get("content") or []:
                p = ct.get("parts") or []
                for part in p:
                    if isinstance(part, str) and part.strip():
                        parts.append(part.strip())
            if not parts:
                continue
            text = "\n".join(parts)
            if author == "user":
                text = clean_user_message(text)
            if text:
                messages.append({"role": author, "text": text})
        if not messages:
            continue
        convs.append(
            {
                "id": c.get("id") or c.get("title", "?"),
                "title": c.get("title", ""),
                "create_time": c.get("create_time", 0.0),
                "messages": messages,
            }
        )
    return convs


def load_codex_corpus() -> list[dict]:
    """Load the raw Codex-session corpus (alternative input for Arch C)."""
    d = json.loads((DATA / "codex_conversations.json").read_text(encoding="utf-8"))
    convs = []
    for c in d["conversations"]:
        msgs = []
        for m in c["messages"]:
            text = m["text"]
            if m["role"] == "user":
                text = clean_user_message(text)
            if text:
                msgs.append({"role": m["role"], "text": text})
        convs.append(
            {
                "id": c["id"],
                "title": c["cwd"],
                "create_time": 0.0,
                "messages": msgs,
            }
        )
    return convs


def conv_to_text(c: dict, max_chars: int = 60000) -> str:
    """Render one conversation as plain text with role prefixes."""
    parts = []
    total = 0
    for m in c["messages"]:
        label = "User" if m["role"] == "user" else "Assistant"
        s = f"{label}: {m['text']}\n"
        if total + len(s) > max_chars:
            break
        parts.append(s)
        total += len(s)
    return "\n".join(parts)


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    if len(text) <= size:
        return [text]
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        out.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return out


def embed(texts: list[str], concurrency: int = 4) -> list[list[float]]:
    """Embed texts with Ollama bge-m3 (1024 dims), batched concurrently."""
    if not OLLAMA_BASE:
        raise RuntimeError("Set OLLAMA_BASE explicitly, or use the offline demo.")
    results: list[list[float] | None] = [None] * len(texts)

    def one(i: int, t: str) -> None:
        for attempt in range(4):
            try:
                body = json.dumps({"model": EMBED_MODEL, "prompt": t}).encode()
                req = urllib.request.Request(
                    f"{OLLAMA_BASE}/api/embeddings", data=body, method="POST"
                )
                req.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(req, timeout=180) as r:
                    results[i] = json.loads(r.read().decode())["embedding"]
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    raise RuntimeError(f"embed failed idx={i}: {exc!r}") from exc
                time.sleep(3 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(lambda p: one(*p), enumerate(texts)))
    return [r for r in results if r is not None]  # type: ignore[misc]


_log_lock = threading.Lock()


def log_call(model: str, endpoint: str, status: str, latency: float, usage: dict | None, err: str = "") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "endpoint": endpoint,
        "status": status,
        "latency_s": round(latency, 1),
        "usage": usage or {},
        "err": err[:300],
    }
    with _log_lock:
        with open(LOG_DIR / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def chat(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 4000,
    temperature: float | None = 0.1,
    retries: int = 3,
) -> tuple[str, dict]:
    """OpenAI-compatible chat completions via the local Sub2API relay."""
    model = model or LLM_MODEL
    if not SUB2API_BASE or not model:
        raise RuntimeError("Set AI_JOB_BASE_URL and LLM_MODEL explicitly for an online run.")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    effort = os.environ.get("AI_JOB_REASONING_EFFORT") or os.environ.get("LLM_REASONING_EFFORT")
    if effort:
        body["reasoning_effort"] = effort
    if temperature is not None:
        body["temperature"] = temperature
    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            data = json.dumps(body).encode("utf-8")
            base_url = SUB2API_BASE.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            req = urllib.request.Request(f"{base_url}/chat/completions", data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "openai-python/1.61.0")
            req.add_header("Authorization", f"Bearer {sub2api_key()}")
            with urllib.request.urlopen(req, timeout=600) as resp:
                r = json.loads(resp.read().decode("utf-8"))
            usage = r.get("usage") or {}
            log_call(model, "chat", "ok", time.time() - t0, usage)
            text = (r["choices"][0].get("message") or {}).get("content") or ""
            return text, usage
        except urllib.error.HTTPError as e:
            try:
                last = f"HTTP {e.code}: {e.read().decode('utf-8')[:200]}"
            except Exception:
                last = f"HTTP {e.code}"
            log_call(model, "chat", f"err{e.code}", time.time() - t0, None, last)
            retryable = e.code in (429, 502, 503) or (
                e.code == 400 and "invalid codex request" in last
            )
            if retryable and attempt < retries:
                time.sleep(20 * (attempt + 1))
                continue
            break
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)[:200]
            log_call(model, "chat", "err", time.time() - t0, None, last)
            if attempt < retries:
                time.sleep(20 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"chat {model} failed: {last}")


def parse_json(text: str):
    """Extract the first JSON value from a model response."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(s)):
            if s[i] == open_ch:
                depth += 1
            elif s[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except Exception:
                        break
    return None


def sqlite_connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.enable_load_extension(False)
    return con


def cosine(a: list[float], b: list[float]) -> float:
    n = len(a)
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
