# PD Chunkfix Baseline for GLM-5.2 MXFP4 on gfx942 (MI325X)

## Overview

Prefill-Decode (PD) disaggregation baseline for GLM-5.2 MXFP4 on 8x MI325X (gfx942)
with MoRI XGMI KV transfer and NEXTN speculative decoding.

## Hardware

- 8x AMD Instinct MI325X (gfx942), 256 GB VRAM per GPU
- Prefill server: GPUs 0-3, TP4
- Decode server: GPUs 4-7, TP4
- KV transfer: MoRI XGMI (8.4 GB/s inter-GPU)

## Model

- GLM-5.2-MXFP4 (GlmMoeDsaForCausalLM)
- 78 layers, 256 experts, 8 per token, DSA attention
- NEXTN speculative decoding: EAGLE, 6 draft tokens, 5 steps
- KV cache dtype: fp8_e4m3

## Scheduler Fixes

Three changes to the prefill scheduler to resolve warm-turn starvation under
concurrent cold-turn surges:

### 1. break to continue in waiting queue loop (scheduler.py)

In `_get_new_batch_prefill_raw`, the waiting queue loop had a `break` after a
failed `add_one_req`. This prevented `init_next_round_input` from being called
on warm turns stuck behind cold turns in the LPM-sorted queue, creating a
deadlock: warm turns could not be prioritized until `init_next_round_input` ran,
but it could not run until the warm turn reached the front of the queue.

Changed `break` to `continue` so the loop continues to the next request,
allowing `init_next_round_input` to be called on warm turns.

### 2. Chunk budget reservation in add_chunked_req (schedule_policy.py)

`add_chunked_req` consumed all `rem_chunk_tokens` (8192) for each cold chunk,
leaving 0 for warm turns. In `add_one_req`, the warm turn entered the truncation
path where `trunc_len = chunk_tokens_limit // page_size * page_size = 0`,
returning `OTHER`.

Added a reservation of 2048 tokens: the cold chunk uses up to 6144, leaving 2048
for warm turns (which need 322-1094 new tokens after cache hit, ceil_paged max
~1152).

### 3. has_chunked_req guard in truncation path (schedule_policy.py)

The `has_chunked_req` parameter in `add_one_req` was unused. Without a guard,
cold turns in the waiting queue could be admitted as new chunked reqs via the
truncation path, consuming the reserved budget and triggering
`assert self.chunked_req is None`.

Added a check: when `has_chunked_req` is True (an active chunked req exists),
return `OTHER` instead of creating a new chunked req. This ensures the reserved
budget is used by warm turns (which fit without truncation).

## Configuration

### Prefill server (cs8k_intlv)

- chunked_prefill_size: 8192
- max_prefill_tokens: 16384
- mem_fraction_static: 0.85
- max_running_requests: 16
- schedule_policy: lpm
- cuda_graph: disabled (prefill and decode)

### Decode server

- mem_fraction_static: 0.80
- max_running_requests: 120
- cuda_graph: full (decode), max_bs 64
- chunked_prefill_size: 65536

## Benchmark Results

### Summary Table (normal_primary, steady_excluding_turn0)

| C | ISL med | OSL med | TTFT (prefill) p50 | TTFT (prefill) p90 | Decode tok/s (warm p50) | Node TPS | TPOT med | Gate |
|---|---------|---------|---------------------|---------------------|--------------------------|----------|----------|------|
| 4 | 153.6K | 312 | 4,530 ms (4.5s) | 6,786 ms (6.8s) | 98.5 | 59.2 | 11.1 ms | PASS |
| 8 | 153.6K | 312 | 4,611 ms (4.6s) | 6,742 ms (6.7s) | 100.6 | 60.9 | 10.6 ms | PASS |

Gates: TTFT warm p90 <= 8s (prefill perspective), decode warm p50 >= 55 tok/s.

### C4 (normal_primary, cs8k_intlv + chunkfix2, 2048 reservation)

Prefill-perspective TTFT (288 warm turns, 0% surge):
- Queue duration: p50 5.8ms, p90 9.9ms, max 2974ms
- Forward duration: p50 4526ms (4.5s), p90 6759ms (6.8s), p99 8125ms (8.1s)
- TTFT (Q+F): p50 4530ms (4.5s), p90 6786ms (6.8s) PASS <=8s

Decode (per active warm request):
- Decode TPS: p50 98.5 tok/s, init-wave p50 126.1 tok/s PASS >=55

Aggregate (320 requests, bench_serving):
- Duration: 1835s (30.6 min), effective concurrency 3.89
- TPOT median: 11.12 ms (89.9 tok/s)
- Accept length: 3.92 (NEXTN)
- Node aggregate: 59.17 tok/s
- Max output tokens/s: 123.0

### C8 (normal_primary, cs8k_intlv + chunkfix2, 2048 reservation)

All 288 warm turns pass (0% surge):
- Queue duration: p50 6ms, p90 9ms, p99 12ms, max 13ms
- Forward duration: p50 4603ms (4.6s), p90 6728ms (6.8s), p99 8053ms (8.1s)
- TTFT (prefill perspective): p50 4.6s, p90 6.7s PASS <=8s

Decode (per active warm request):
- Decode TPS: p50 100.6 tok/s, init-wave p50 133.3 tok/s PASS >=55

Aggregate (320 requests, bench_serving):
- Duration: 1784s (29.7 min), effective concurrency 7.29
- TPOT median: 10.60 ms (94.3 tok/s)
- Accept length: 3.83 (NEXTN)
- Node aggregate: 60.88 tok/s
- Max output tokens/s: 124.0

Fix progression at C8:
- No fix: 288/288 warm turns had 27-82s queue (FAIL)
- 1024 reservation: 146/288 = 50.7% surge at 33-85s queue (FAIL)
- 2048 reservation: 0/288 surge, all pass (PASS)

Note: bench_serving TTFT includes concurrency wait from all-requests-at-once
methodology. The prefill-perspective TTFT (queue + forward from server logs)
is the real interactive TTFT and is the gate metric.

## Root Cause Analysis

The bottleneck at C8 is the prefill scheduler chunk budget allocation during
the cold turn surge, NOT prefill throughput, KV transfer, or decode throughput.

Evidence:
1. Warm turn forward_duration: 4.5-6.8s p50-p90 (includes cold chunk batch)
2. KV usage: 7-21% (plenty of capacity)
3. MoRI transfer: 5-9 GB/s at steady state
4. Queue durations are bimodal: 1-5 ms steady, 33-85 s surge (without fix)
5. The break-to-continue fix drops steady-state queue from 27-82 s to 1-5 ms
6. Decode TPS per warm request: 98-133 tok/s (well above 55 gate)

## Configuration Details

### Prefill server (cs8k_intlv)

- chunked_prefill_size: 8192
- max_prefill_tokens: 16384
- mem_fraction_static: 0.85
- max_running_requests: 16
- schedule_policy: lpm
- cuda_graph: disabled (prefill and decode)

### Decode server

- mem_fraction_static: 0.80
- max_running_requests: 120
- cuda_graph: full (decode), max_bs 64
- chunked_prefill_size: 65536

## Next Steps

- C16-C32: KV capacity becomes the constraint (~39 GB KV per user per GPU,
  ~180 GB available = ~4-5 users per GPU). Need LayerSplit on decode.
- Prefill throughput: CP for faster cold turn processing
- Dead ends: PP2xTP2 (gloo TCP crash), DCP (CUDA-only), MoE-EP (AITER fp4x2
  crash), Mooncake TCP (1.5-2.3 GB/s vs MoRI 8.4 GB/s)
