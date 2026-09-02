# GLM-5.3-Flash-MXFP4 on MI355X (gfx950)

Baseline deployment recipe, reproduction scripts, and measured numbers for
[OneNexus/GLM-5.3-Flash-MXFP4](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4)
(`Glm5NextForConditionalGeneration`, `glm5_next`, 212 GiB) served with TP4/EP4
on four AMD Instinct MI355X.

## The overlay is required

`glm5_next` is **not** in upstream SGLang and not in any published
`lmsysorg/sglang-rocm` image — both `v0.5.18-rocm724-mi35x-20260822` and
`-20260902` contain zero references to `Glm5Next`/`glm5_next`, and upstream
`main` has no such model file. The candidate-v30 overlay mounted over
`python/sglang` *is* the implementation (25 files reference the architecture),
not a tuning patch on top of a working model.

Consequences:

- `serve.sh` requires `OVERLAY_DIR`. Without it the server cannot start.
- A newer image cannot substitute for the overlay, and because the overlay
  replaces `python/sglang` wholesale, a newer image's SGLang code never
  executes. The image supplies AITER/Triton/ROCm only.
- Rebasing the overlay onto a newer image is a port, not a cherry-pick:
  upstream restructured this area (`dsa_indexer_kpool.py` is gone, replaced by
  `kpool_fp8_index.py` / `kpool_plan.py` / `paged_mqa_logits_backend.py`), so of
  the overlay's 48 changed files, 1 applies cleanly, 44 conflict, and 3 no
  longer exist upstream.

## Launch

```bash
hf download OneNexus/GLM-5.3-Flash-MXFP4 \
    --revision 21e1124f735fd7b7836189d6c13d5eedfef3fb88 \
    --local-dir /data/models/GLM-5.3-Flash-MXFP4

MODEL_DIR=/data/models/GLM-5.3-Flash-MXFP4 \
OVERLAY_DIR=/path/to/candidate-v30/sglang \
AITER_JIT_DIR=/path/to/aiter-jit-cache \
SGLANG_API_KEY=... ./serve.sh
```

## Verify

```bash
SGLANG_API_KEY=... python3 probe.py --port 30037
```

Reasoning and tool calling both parse: responses carry `reasoning_content`, and
a request with `tools` returns OpenAI `tool_calls`.

## Measured baseline

AgentX smoke via the `sglang-agentx-benchmark` skill — pinned corpus
`semianalysis_cc_traces_weka_062126`, `--scenario inferencex-agentx-mvp`, seed
42, 10 traces, c8, 900 s, no `--max-context-length`. Result:
`submission_valid: true`, 269 requests, 0 errors.

| Metric | avg | p50 | p90 | p99 |
| --- | --- | --- | --- | --- |
| TTFT (ms) | 1047 | 565 | 2077 | 7245 |
| ITL (ms) | 3.21 | 2.92 | 4.66 | 7.06 |
| Request latency (ms) | 3570 | 1811 | 7039 | 23152 |
| Output tok/s/user | 536.0 | 341.9 | — | — |

Aggregate 37,477 tok/s (37,204 input), 98.05% theoretical prefix-cache hit,
effective concurrency 1.03 — Flash retires work fast enough that the same
8-lane replay leaves fewer requests in flight, so these latencies were measured
under lighter queueing than the full GLM-5.3 baseline.

MTP/EAGLE at 5/1/6 accepts **4.58 tokens per verify step** (p50 4.80, ceiling
6; accept rate 0.72) over 1061 decode batches, at 409 tok/s mean generation.
Draft utilisation is 76%, materially better than full GLM-5.3's 64% on the same
shape — Flash's MTP head predicts its own output well, which is most of why its
ITL is ~2.8x lower.

Smoke mode is 10 of 393 traces, so `submission_valid: true` means rule-
compliant, not comparable to InferenceX CI. Use `MODE=full ./run_agentx.sh`.

The bundled GSM8K gate scores 18/20 because two of its items are defective (q8's
key says 140 where (60+120+360)/3 = 180; q9 asks for two-thirds of 25).
`run_agentx.sh` gates at 0.90 for that reason.

## DSA logits limit (ROCm) — `patches/`

On ROCm the DSA indexer's `[num_q x num_k]` fp32 MQA-logits tensor goes to
aiter's `fp8_mqa_logits`, which only compiles below 2 GiB; above it the fallback
fails to compile and `abort()`s every TP rank. Upstream fixed this for the
`Indexer` class in sgl-project/sglang#36960.

This model does not use that class. Its `text_config` declares
`index_kpool: 4`, and `deepseek_v2.py` selects
`IndexerKPool if get_dsa_index_kpool(config) > 1 else Indexer`, so #36960 does
not cover it — the subject of sgl-project/sglang#37478, still open.

The overlay already bounds the ragged path with `_mqa_logits_chunk_rows`, which
is stricter than upstream's fix because it accounts for aiter padding the logits
row stride to 256 elements. `patches/0001-bound-kpool-cp-mqa-logits.patch`
extends that same helper to the one call site that lacked it,
`_get_topk_ragged_with_cp`.

That path is inert unless context parallelism is enabled
(`enable_dsa_prefill_context_parallel`, `dcp_size > 1`), which this baseline
does not use — the patch is pre-emptive. With CP on and
`--chunked-prefill-size C`, the unpatched wall sits at `2^31 / C` tokens of
context, because kpool's `num_k = kv_len / index_kpool` divides by 4 and cancels
fp32's 4 bytes: 65,536 tokens at C=32768, 131,072 at C=16384. Agentic contexts
of 79K-120K exceed the first of those.

## Files

| File | Purpose |
| --- | --- |
| `serve.sh` | the baseline launch (requires the overlay) |
| `probe.py` | correctness + decode + long-prefill smoke |
| `run_agentx.sh` | AgentX replay, canonical flags |
| `patches/0001-bound-kpool-cp-mqa-logits.patch` | bounds the kpool CP MQA-logits path (#37478) |
