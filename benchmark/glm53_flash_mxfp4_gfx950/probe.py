#!/usr/bin/env python3
"""Smoke probe for the GLM-5.3-Flash-MXFP4 gfx950 baseline.

Checks correctness on four well-posed arithmetic items, decode throughput at a
fixed output length, and a long-prefill request. The long prompt is salted per
run: an identical prompt is served from the radix/HiCache prefix cache and
measures the cache, not the model.

  SGLANG_API_KEY=... python3 probe.py --port 30037 --model OneNexus/GLM-5.3-Flash-MXFP4
"""

import argparse
import json
import os
import random
import time
import urllib.request

QA = [
    ("What is 17 times 19? Reply with just the number.", "323"),
    (
        "A program had 60 downloads in month 1. Month 2 had twice month 1. "
        "Month 3 had 3 times month 2. What is the average over three months? Just the number.",
        "180",
    ),
    ("If a train travels 240 km in 3 hours, what is its speed in km/h? Just the number.", "80"),
    ("What is 15% of 340? Just the number.", "51"),
]


def post(host, port, key, payload, timeout):
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()), time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="30037")
    ap.add_argument("--model", default="OneNexus/GLM-5.3-Flash-MXFP4")
    ap.add_argument("--prefill-words", type=int, default=60000)
    args = ap.parse_args()
    key = os.environ["SGLANG_API_KEY"]

    correct = 0
    for question, answer in QA:
        body, _ = post(args.host, args.port, key,
                       {"model": args.model, "temperature": 0, "max_tokens": 3000,
                        "messages": [{"role": "user", "content": question}]}, 300)
        if answer in (body["choices"][0]["message"].get("content") or ""):
            correct += 1
    print(f"correctness      {correct}/{len(QA)}")

    body, elapsed = post(args.host, args.port, key,
                         {"model": args.model, "temperature": 0, "max_tokens": 800,
                          "ignore_eos": True,
                          "messages": [{"role": "user",
                                        "content": "Explain how MoE expert routing works, step by step."}]}, 600)
    produced = body["usage"]["completion_tokens"]
    print(f"decode           {produced} tokens in {elapsed:.2f}s = {produced / elapsed:.1f} tok/s")

    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    salt = f"{os.getpid()}x{random.randrange(10 ** 9)}"
    filler = " ".join(
        (f"{words[i % 10]}{i}u{salt}" if i % 89 == 0 else f"{words[i % 10]}{i}")
        for i in range(args.prefill_words)
    )
    body, elapsed = post(args.host, args.port, key,
                         {"model": args.model, "temperature": 0, "max_tokens": 16,
                          "messages": [{"role": "user", "content": "Reply OK.\n\n" + filler}]}, 900)
    prompt_tokens = body["usage"]["prompt_tokens"]
    print(f"long prefill     {prompt_tokens} tokens in {elapsed:.2f}s = {prompt_tokens / elapsed:.0f} tok/s")


if __name__ == "__main__":
    main()
