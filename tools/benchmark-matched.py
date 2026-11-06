#!/usr/bin/env python3
"""Replay synthetic historical cases against an OpenAI-compatible SSE endpoint.

Supply MODEL_BASE_URL and MODEL_API_KEY through the environment. Results never
include either value. This bounded decode test does not evaluate code correctness.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

TZ = datetime.timezone(datetime.timedelta(hours=8))


def collect(lines, start, clock=time.monotonic):
    text, first, last, usage, finish, done, reasoning = "", None, None, {}, None, False, 0
    for line in lines:
        if not line.startswith(b"data: "):
            continue
        if line.strip() == b"data: [DONE]":
            done = True
            break
        obj = json.loads(line[6:])
        usage = obj.get("usage") or usage
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content") or ""
            reasoning += len(delta.get("reasoning_content") or "")
            if content:
                now = clock()
                first = now if first is None else first
                last = now
                text += content
            finish = choice.get("finish_reason") or finish
    total = clock() - start
    tokens = usage.get("completion_tokens", 0)
    decode = (tokens - 1) / (last - first) if first is not None and last > first and tokens > 1 else None
    return dict(done=done, finish_reason=finish, usage=usage,
                ttft_seconds=None if first is None else first - start,
                last_token_seconds=None if last is None else last - start,
                total_seconds=total, decode_tokens_per_second=decode,
                output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                output_chars=len(text), reasoning_chars=reasoning)


def matched(result, case):
    usage = result["usage"]
    return bool(result["done"] and result["finish_reason"] == "length"
                and usage.get("prompt_tokens") == case["expected_prompt_tokens"]
                and usage.get("completion_tokens") == case["max_tokens"]
                and usage.get("completion_tokens_details", {}).get("reasoning_tokens") == 0
                and usage.get("prompt_tokens_details", {}).get("cached_tokens") == 0
                and result["reasoning_chars"] == 0
                and result["decode_tokens_per_second"] is not None)


def run_case(base, key, case):
    digest = hashlib.sha256(json.dumps(case["messages"]).encode()).hexdigest()
    if digest != case["prompt_sha256"]:
        raise ValueError("Prompt SHA256 mismatch")
    body = dict(model="deepseek-chat", messages=case["messages"], max_tokens=case["max_tokens"],
                temperature=0, top_p=1, stream=True, stream_options={"include_usage": True},
                thinking={"type": "disabled"}, chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        result = collect(response, start)
        result["http_status"] = response.status
    result.update(scenario=case["scenario"], trial=case["trial"], prompt_sha256=digest,
                  max_tokens=case["max_tokens"], measured_at=datetime.datetime.now(TZ).isoformat())
    result["matched"] = matched(result, case)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=Path(__file__).parents[1] / "examples/matched-speed-cases.json")
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cases = json.loads(args.cases.read_text())
    report = dict(label=args.label, measured_at=datetime.datetime.now(TZ).isoformat(),
                  method="(completion_tokens - 1) / (last text event - first text event)", results=[])
    for case in cases:
        result = run_case(os.environ["MODEL_BASE_URL"], os.environ["MODEL_API_KEY"], case)
        report["results"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    report["matched"] = all(r["matched"] for r in report["results"])
    report["summaries"] = {}
    for scenario in sorted({r["scenario"] for r in report["results"]}):
        runs = [r for r in report["results"] if r["scenario"] == scenario]
        if all(r["matched"] for r in runs):
            report["summaries"][scenario] = {k: statistics.median(r[k] for r in runs) for k in (
                "decode_tokens_per_second", "ttft_seconds", "total_seconds")}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["matched"]:
        raise SystemExit("Some requests were not comparable; inspect per-run results")


if __name__ == "__main__":
    main()
