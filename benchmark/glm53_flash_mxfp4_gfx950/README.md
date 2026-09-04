# GLM-5.3-Flash-MXFP4 on MI355X (gfx950)

Baseline deployment recipe, reproduction scripts, and measured numbers for
[OneNexus/GLM-5.3-Flash-MXFP4](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4)
(`Glm5NextForConditionalGeneration`, `glm5_next`, 212 GiB) served with TP4/EP4
on four AMD Instinct MI355X.

## The overlay is required

`glm5_next` is **not** in upstream SGLang. The
`v0.5.18-rocm724-mi35x-20260822` and `-20260902` images contain zero references
to `Glm5Next`/`glm5_next`, while the `-20260903` image still lacks
`Glm5NextProcessor` and `Glm5NextImageProcessor`. The candidate-v30 overlay mounted over
`python/sglang` *is* the implementation (25 files reference the architecture),
not a tuning patch on top of a working model.

Consequences:

- `serve.sh` requires `OVERLAY_DIR`. Without it the server cannot start.
- A newer image cannot substitute for the overlay, and because the overlay
  replaces `python/sglang` wholesale, a newer image's SGLang code never
  executes. The image supplies AITER/Triton/ROCm only.
- Multimodal input additionally requires the pinned Hugging Face Transformers
  processor checkout and the pinned official GLM-5.3-Flash chat template.
  Without the processor, image requests return HTTP 200 but the image pixels
  never enter the prompt. The OneNexus model revision carries the immediately
  previous Z.ai template, so the preparation step verifies and installs the
  current template explicitly.
- Rebasing the overlay onto a newer image is a port, not a cherry-pick:
  upstream restructured this area (`dsa_indexer_kpool.py` is gone, replaced by
  `kpool_fp8_index.py` / `kpool_plan.py` / `paged_mqa_logits_backend.py`), so of
  the overlay's 48 changed files, 1 applies cleanly, 44 conflict, and 3 no
  longer exist upstream.

## Prepare the overlay

This fixing branch carries the cache-integrity changes in two forms:

- the native upstream-tree implementation under `python/sglang`; and
- `patches/0002-mamba-radix-finished-state-integrity.patch`, adapted to the
  candidate-v30 overlay that the deployment actually imports.

The overlay patch is based on candidate-v30 source commit
`62b9a4a1c8fa9f84db0c39d518c7cde156ecb3a9`. Prepare a separate copy so the
known-good source remains untouched:

```bash
BASE_OVERLAY_DIR=/path/to/candidate-v30/python/sglang \
OUTPUT_DIR=/path/to/candidate-v30-mamba-fixed/python/sglang \
./prepare_overlay.sh
```

The script rejects an incompatible patch or an existing output path, applies
the patch, compiles the resulting Python tree, and verifies both integrity
guards are present. The speculative-overshoot/checkpoint fix addresses a
confirmed reachable cache-integrity defect. The freed-overlap-row guard is
defensive hardening; instrumentation has not established that path as the
specific Meridian trigger.

Prepare the processor stack separately. The Transformers commit, Tokenizers
version, official template revision, and template digest are pinned, and the
script refuses to overwrite either destination:

```bash
TRANSFORMERS_DIR=/data/runtime/transformers-e4052f55 \
TRANSFORMERS_RUNTIME_DIR=/data/runtime/transformers-deps-e4052f55 \
./prepare_transformers.sh
```

## Launch

```bash
hf download OneNexus/GLM-5.3-Flash-MXFP4 \
    --revision 21e1124f735fd7b7836189d6c13d5eedfef3fb88 \
    --local-dir /data/models/GLM-5.3-Flash-MXFP4

MODEL_DIR=/data/models/GLM-5.3-Flash-MXFP4 \
OVERLAY_DIR=/path/to/candidate-v30-mamba-fixed/python/sglang \
AITER_JIT_DIR=/path/to/aiter-jit-cache \
TRANSFORMERS_DIR=/data/runtime/transformers-e4052f55 \
TRANSFORMERS_RUNTIME_DIR=/data/runtime/transformers-deps-e4052f55 \
SGLANG_API_KEY=... ./serve.sh
```

The defaults keep the production performance recipe intact: EAGLE 5/1/6,
TP4/EP4, TileLang DSA, AITER MoE, FP8 KV, radix cache, and HiCache
write-through/direct all remain enabled. GPUs 4-7 default to NUMA node 1 with
CPUs 96-191; override `GPUS`, `CPUSET_CPUS`, and `CPUSET_MEMS` together on a
different topology. CPU multimodal feature transport is explicit because the
GPU-resident transports are CUDA-only; it does not disable the vision encoder.

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
| `prepare_overlay.sh` | copies and patches candidate-v30 without changing the source tree |
| `prepare_transformers.sh` | prepares the pinned GLM-5.3 multimodal processor stack |
| `serve.sh` | the baseline launch (requires the patched overlay) |
| `probe.py` | correctness + decode + long-prefill smoke |
| `run_agentx.sh` | AgentX replay, canonical flags |
| `patches/0001-bound-kpool-cp-mqa-logits.patch` | bounds the kpool CP MQA-logits path (#37478) |
| `patches/0002-mamba-radix-finished-state-integrity.patch` | ports the finished-request Mamba/radix integrity fixes to candidate-v30 |
