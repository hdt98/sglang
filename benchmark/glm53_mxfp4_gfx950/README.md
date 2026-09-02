# GLM-5.3-MXFP4 on MI355X (gfx950)

Baseline deployment recipe, reproduction scripts, and measured numbers for
[OneNexus/GLM-5.3-MXFP4](https://huggingface.co/OneNexus/GLM-5.3-MXFP4) — an
OCP MXFP4 E2M1 quantization of the full GLM-5.3 (`GlmMoeDsaForCausalLM`, 282
shards, 407.92 GiB) — served with TP4/EP4 on four AMD Instinct MI355X.

The recipe is the model card's stock InferenceX/SGLang path. Only two things
differ, both for a shared endpoint rather than the card's single-user latency
cell:

| Flag | Model card | Here | Why |
| --- | --- | --- | --- |
| `--max-running-requests` | 2 | 64 | the card's value is a c2 latency cell |
| `--cuda-graph-max-bs` | 2 | 64 | same |
| `--api-key` | absent | required | the endpoint is reachable off-host |

## Launch

```bash
hf download OneNexus/GLM-5.3-MXFP4 \
    --revision 104690ed94d48341ec9de43b1bc12d30f7eaa86e \
    --local-dir /data/models/GLM-5.3-MXFP4

MODEL_DIR=/data/models/GLM-5.3-MXFP4 SGLANG_API_KEY=... ./serve.sh
```

Weights load in roughly 4.5 minutes. `serve.sh` defaults to the
`v0.5.18-rocm724-mi35x-20260902` image; see *DSA logits limit* below for why
that tag is the floor.

## Verify

```bash
SGLANG_API_KEY=... python3 probe.py --port 30000
```

Four arithmetic items, a fixed-length decode, and a long prefill. Expected on
this hardware: `4/4`, ~155–175 tok/s decode, and a 227K-token prefill that
completes rather than aborting.

Reasoning and tool calling both parse (`--reasoning-parser glm45`,
`--tool-call-parser glm47`): responses carry `reasoning_content`, and a request
with `tools` returns OpenAI `tool_calls` with `finish_reason: tool_calls`.

## Measured baseline

AgentX smoke via the `sglang-agentx-benchmark` skill — pinned corpus
`semianalysis_cc_traces_weka_062126`, `--scenario inferencex-agentx-mvp`, seed
42, 10 traces, c8, 900 s, no `--max-context-length`. Result:
`submission_valid: true`, 224 requests, 0 errors.

| Metric | avg | p50 | p90 | p99 |
| --- | --- | --- | --- | --- |
| TTFT (ms) | 4010 | 3329 | 8597 | 15207 |
| ITL (ms) | 9.06 | 8.18 | 12.79 | 30.60 |
| Request latency (ms) | 7904 | 3539 | 20152 | 59307 |
| Output tok/s/user | 127.0 | 122.3 | 185.0 | 246.1 |

Aggregate 31,039 tok/s (30,808 input), 98.1% theoretical prefix-cache hit.

MTP/EAGLE at 5/1/6 accepts **3.82 tokens per verify step** (p50 3.77, ceiling
6; accept rate 0.56) over 763 decode batches, at 260 tok/s mean generation.
Draft utilisation is 64%, so the 5th and 6th draft tokens are into diminishing
returns — 4/1/5 is worth measuring. For reference, EAGLE 3/1/4 on the same
hardware accepted 3.11 of 4 (78% utilisation) and was slower on every latency
percentile.

Smoke mode is 10 of 393 traces: `submission_valid: true` means the run obeyed
AgentX's locked rules, not that it is comparable to InferenceX CI. Use
`MODE=full ./run_agentx.sh` for that.

The bundled GSM8K gate scores 18/20 because two of its items are defective (q8's
key says 140 where (60+120+360)/3 = 180, which the model answers correctly; q9
asks for two-thirds of 25). `run_agentx.sh` gates at 0.90 for that reason.

## DSA logits limit (ROCm)

The DSA indexer materialises an `[num_q x num_k]` fp32 MQA-logits tensor. On
ROCm that goes to aiter's `fp8_mqa_logits`, which only compiles below 2 GiB;
above it the fallback fails to compile and `abort()`s **every TP rank**, taking
the server down mid-request. On MI355X the memory-based budget returns ~4.4 GB,
so it never binds first.

sgl-project/sglang#36960 caps the budget at the kernel limit. Images before
`v0.5.18-rocm724-mi35x-20260902` do not carry it: with
`--chunked-prefill-size 32768` this reproduces at roughly 16K tokens of
context, and a 79K-token agentic request aborts the pool. Verified fixed here
with a 227K-token prefill.

One residual gap: the upstream cap bounds raw `numel()`, while aiter pads the
logits row stride to 256 elements, so shapes where the padded tensor crosses
2 GiB but the raw one does not still abort. The band is narrow (~0.2% of
shapes) but reachable, since a chunk's row count is arbitrary.

## Files

| File | Purpose |
| --- | --- |
| `serve.sh` | the baseline launch |
| `probe.py` | correctness + decode + long-prefill smoke |
| `run_agentx.sh` | AgentX replay, canonical flags |
