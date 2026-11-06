#!/usr/bin/env python3
"""Bounded real-pair acceptance. MODEL_BASE_URL / MODEL_API_KEY stay in the environment.

Run smoke, long, then resume after restarting the pair. Synthetic prompts only;
the JSON report contains timings, usage and hashes, never credentials or prompts.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time
import urllib.parse
import uuid
import zlib

import requests


BASE = os.environ.get("MODEL_BASE_URL", "").rstrip("/")
KEY = os.environ.get("MODEL_API_KEY", "")
TZ = datetime.timezone(datetime.timedelta(hours=8))


def request(path, payload=None, auth=True, stream=False):
    headers = {"Authorization": "Bearer " + KEY} if auth else {}
    return requests.request("GET" if payload is None else "POST", BASE + path,
                            json=payload, headers=headers, stream=stream, timeout=(10, 900))


def events(response):
    response.raise_for_status()
    for line in response.iter_lines(chunk_size=1):
        if line.startswith(b"data: "):
            data = line[6:]
            yield "[DONE]" if data == b"[DONE]" else json.loads(data)


def body(messages, max_tokens=32, **extra):
    return dict(model="deepseek-chat", messages=messages, max_tokens=max_tokens,
                temperature=0, chat_template_kwargs={"enable_thinking": False}, stream=True,
                stream_options={"include_usage": True}, **extra)


def chat(messages, max_tokens=32, **extra):
    start = time.monotonic()
    first, text, reasoning, usage, calls, finish, done = None, "", "", {}, {}, None, False
    with request("/chat/completions", body(messages, max_tokens, **extra), stream=True) as r:
        for ev in events(r):
            if ev == "[DONE]":
                done = True
                break
            usage = ev.get("usage") or usage
            for choice in ev.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    first = first if first is not None else time.monotonic() - start
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                for c in delta.get("tool_calls", []):
                    item = calls.setdefault(c["index"], {"name": "", "arguments": "", "pieces": 0})
                    f = c.get("function", {})
                    item["name"] += f.get("name") or ""
                    if f.get("arguments"):
                        item["arguments"] += f["arguments"]
                        item["pieces"] += 1
                finish = choice.get("finish_reason") or finish
    assert done and finish, "Incomplete SSE response"
    assert not reasoning, "This acceptance request must not spend its budget on reasoning"
    return dict(seconds=time.monotonic() - start, first_seconds=first, usage=usage,
                text=text, reasoning=reasoning, calls=list(calls.values()), finish=finish, done=done)


def summary(r):
    return {**{k: v for k, v in r.items() if k not in ("text", "reasoning", "calls")},
            "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
            "text_chars": len(r["text"]), "reasoning_chars": len(r["reasoning"])}


def smoke():
    with request("/models", auth=False) as r:
        assert r.status_code == 401, r.status_code
    with request("/models") as r:
        r.raise_for_status()
        assert any(x["id"] == "deepseek-chat" for x in r.json()["data"])
    ordinary = chat([{"role": "user", "content": "Reply exactly PAIR_READY."}])
    assert "PAIR_READY" in ordinary["text"], ordinary["text"]
    literal = 'literal </tool_call> <tool_call> </invoke> </｜DSML｜invoke> <｜DSML｜parameter> & "quoted"\nsecond line'
    tool = {"type": "function", "function": {"name": "capture", "description": "Capture the exact supplied text",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}
    called = chat([{"role": "user", "content": "Call capture with text equal to this JSON string: " + json.dumps(literal)}],
                  256, tools=[tool], tool_choice={"type": "function", "function": {"name": "capture"}})
    assert called["finish"] == "tool_calls" and len(called["calls"]) == 1, called
    call = called["calls"][0]
    assert call["name"] == "capture" and json.loads(call["arguments"])["text"] == literal, call
    assert call["pieces"] > 1, "Tool arguments were not streamed in pieces"
    with request("/responses", {"model": "deepseek-chat", "input": "Reply exactly RESPONSES_READY.",
                  "max_output_tokens": 256, "include": ["reasoning.encrypted_content"], "stream": True,
                  "reasoning": {"effort": "none"}}, stream=True) as r:
        evs = list(events(r))
    completed = [e for e in evs if isinstance(e, dict) and e.get("type") == "response.completed"]
    assert len(completed) == 1, "Responses did not complete"
    response = completed[0]["response"]
    assert "RESPONSES_READY" in json.dumps(response["output"]), response
    with request("/messages", {"model": "deepseek-chat", "messages": [{"role": "user", "content": "Reply ANTHROPIC_READY."}],
                  "max_tokens": 512, "thinking": {"type": "enabled", "budget_tokens": 128}}) as r:
        r.raise_for_status()
        anthropic = r.json()
    assert any(x["type"] == "thinking" and x.get("thinking") for x in anthropic["content"]), anthropic
    assert any(x["type"] == "text" and "ANTHROPIC_READY" in x.get("text", "") for x in anthropic["content"]), anthropic
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64)) + chunk(b"IEND", b"")
    vision = chat([{"role": "user", "content": [{"type": "text", "text": "What is the solid color? Answer one word."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}], 32)
    assert "red" in vision["text"].lower(), vision["text"]
    return {"ordinary": summary(ordinary), "tool": {"pieces": call["pieces"], "exact_arguments": True},
            "responses_include": True, "responses_events": len(evs), "anthropic_thinking": True,
            "vision": summary(vision), "unauthenticated_status": 401}


def long_messages(case, target=300000, nonce="stable-restart-fixture"):
    # Distinct early prefixes prevent the two sessions from borrowing one another's cache.
    unit = f" {case} 314159 271828 161803 record.\n"
    prefix = f"Case {case}; fixture {nonce}.\n"
    short = [{"role": "user", "content": prefix + unit * 1000 + f"\nReply only {case}_READY."}]
    with request("/tokenize", {"messages": short, "chat_template_kwargs": {"enable_thinking": False}}) as r:
        r.raise_for_status()
        count = r.json()["count"]
    repeats = int(1000 * target / count)
    return [{"role": "user", "content": prefix + unit * repeats + f"\nReply only {case}_READY."}]


def cached(result):
    return result["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)


def long_sessions(resume=False):
    a, b = long_messages("ALPHA"), long_messages("BETA")
    if resume:
        result = chat(a)
        assert cached(result) > 280000, result["usage"]
        assert "ALPHA_READY" in result["text"], result["text"]
        # The operator must pair this with both ranks' restart records and disk-restore logs.
        return {"cache_reuse_after_external_restart": summary(result)}
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(chat, a)
        time.sleep(4)
        short = chat([{"role": "user", "content": "Reply SHORT_CUT_IN_READY."}])
        short_done = time.monotonic() - start
        first = future.result(timeout=900)
    assert short_done < first["first_seconds"], {"short_done": short_done, "long_first": first["first_seconds"]}
    second = chat(b)
    again_a, again_b = chat(a), chat(b)
    assert "SHORT_CUT_IN_READY" in short["text"], short["text"]
    for result, word in ((first, "ALPHA_READY"), (again_a, "ALPHA_READY"),
                         (second, "BETA_READY"), (again_b, "BETA_READY")):
        assert word in result["text"], result["text"]
    for result in (first, second):
        assert 280000 <= result["usage"]["prompt_tokens"] <= 320000, result["usage"]
    assert cached(again_a) > 280000 and cached(again_b) > 280000
    return {"alpha_first": summary(first), "beta_first": summary(second), "alpha_again": summary(again_a),
            "beta_again": summary(again_b), "short": summary(short), "short_done_seconds": short_done}


def progress(cancel_after):
    nonce = uuid.uuid4().hex
    messages = long_messages("GAMMA", 300000, nonce)
    target = urllib.parse.urlsplit(BASE)
    assert target.scheme == "http", "Raw cancellation fixture is restricted to the LAN HTTP endpoint"
    data = json.dumps(body(messages, 32)).encode()
    headers = (f"POST {target.path}/chat/completions HTTP/1.1\r\nHost: {target.netloc}\r\n"
               f"Authorization: Bearer {KEY}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n").encode()
    with socket.create_connection((target.hostname, target.port or 80), timeout=10) as sock:
        sock.sendall(headers + data)
        deadline = time.monotonic() + cancel_after
        received = bytearray()
        sock.settimeout(1)
        while time.monotonic() < deadline:
            try:
                piece = sock.recv(65536)
            except socket.timeout:
                continue
            if not piece:
                raise AssertionError("Original stream ended before cancellation")
            received.extend(piece)
            assert len(received) < 1 << 20, "Original response exceeded fixture bound"
        assert received.startswith((b"HTTP/1.1 200", b"HTTP/1.0 200")), "Original request was not accepted"
        for line in bytes(received).splitlines():
            if line.startswith(b"data: "):
                assert line[6:] != b"[DONE]", "Original request completed before cancellation"
                ev = json.loads(line[6:])
                for choice in ev.get("choices", []):
                    delta = choice.get("delta", {})
                    assert not any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")), "Prefill already finished"
        sock.shutdown(socket.SHUT_RDWR)
    time.sleep(3)
    retry = chat(messages)
    assert "GAMMA_READY" in retry["text"], retry["text"]
    reused = cached(retry)
    assert 65536 <= reused < 280000, {"cached": reused, "usage": retry["usage"]}
    return {"cancel_after_seconds": cancel_after, "original_http_status": 200,
            "fixture_nonce": nonce, "prompt_sha256": hashlib.sha256(data).hexdigest(),
            "retry": summary(retry), "progress_cached_tokens": reused}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("smoke", "long", "resume", "progress"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cancel-after", type=float, default=30)
    args = p.parse_args()
    if not BASE or not KEY:
        p.error("MODEL_BASE_URL and MODEL_API_KEY are required")
    result = {"measured_at": datetime.datetime.now(TZ).isoformat(), "phase": args.phase}
    if args.phase == "smoke":
        result["result"] = smoke()
    elif args.phase == "progress":
        result["result"] = progress(args.cancel_after)
    else:
        result["result"] = long_sessions(args.phase == "resume")
    result["passed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
