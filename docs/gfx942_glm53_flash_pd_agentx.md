# GLM-5.3 Flash PD on gfx942: AgentX frontier report

This document tracks the trace-led optimization of GLM-5.3 Flash FP8
prefill-decode disaggregation on one 8x MI325X node. Results are promotion
evidence only when they use the canonical, uncapped AgentX corpus and report
`submission_valid=true`.

## Contract

- Dataset: `semianalysis_cc_traces_weka_062126`
- Full corpus: 393 traces, with no `--max-context-length`
- Seed: 42
- Trajectory start ratio: 0.25-0.75
- Trace idle-gap cap: 300 seconds
- Warmup: 10 requests per lane
- Failed-request threshold: 0.10
- Smoke: C8, 10 traces, 900 seconds
- Full: all traces, 1,800 seconds at each tested concurrency

A point passes only when all four gates pass:

| Metric | Gate |
|---|---:|
| TTFT p50 | <= 5.0 s |
| TTFT p90 | <= 11.0 s |
| Decode throughput/user p10 | >= 34 tok/s |
| Decode throughput/user p50 | >= 55 tok/s |

`Decode throughput/user` is the gating throughput metric. End-to-end output
throughput, which includes TTFT, is reported only as auxiliary context and is
never substituted for either decode-throughput gate.

## Runtime invariants

- TP4 prefill on GPUs 0-3 and TP4 decode on GPUs 4-7 for the reference cell
- MoRI XGMI KV transfer
- EAGLE speculative token decoding on the decode role only, with adaptive
  5/1/6 gears. The prefill role runs the matched draft-state prefill path so it
  can hand off draft KV, hidden state, first-token probabilities, and DSA seed
  indices; it does not decode output tokens.
- Full target-model decode graphs
- Decode radix cache enabled for the hybrid KDA model
- FP8 attention KV cache
- Role-local KDA sizing and tuning
- `sglang-agentx-benchmark` is the only workload used for frontier claims

## Best verified point

The best verified point is still the r22 canonical smoke at C8. It passes all
four gates, but a smoke is not a full-corpus frontier claim.

Artifacts:

- Server: `/scratch/sonle5/pd_runs/pd_glm53_pr36607_r22_longctx_qp8`
- AgentX: `/scratch/sonle5/pd_runs/agentx_pr36607_r22_longctx_qp8_smoke_c8`

| Metric | r22 C8 |
|---|---:|
| Submission valid | true |
| Warmup / profiling requests | 87/87 / 214/214 |
| Errors | 0 |
| TTFT p50 / p90 | 2.459 / 3.185 s |
| Decode throughput/user p10 / p50 | 93.43 / 131.36 tok/s |
| Node output throughput | 214.33 tok/s |
| Input/prefill throughput | 29,892.83 tok/s |
| ITL p50 / p90 / average | 7.60 / 10.70 / 8.25 ms |
| ISL average / p50 / p90 / max | 129,908 / 113,880 / 252,098 / 485,131 |
| OSL average / p50 / p90 / max | 931 / 272 / 2,170 / 18,302 |
| Effective concurrency average / p90 / max | 2.03 / 4 / 6 |
| Theoretical / observed cache hit | 98.14% / 98.04% |
| KV / KDA peak usage | 15% / 2% |
| Retractions | 0 |

Startup allocation confirms that r22 is not KV- or KDA-capacity-bound at C8.
On each prefill rank, the KDA/Mamba pool admits 714 states and consumes 0.86 GB
of convolution state plus 24.44 GB of SSM state. The target KV pool holds
14,175,552 tokens in 93.52 GB and the matched draft KV pool holds the same token
count in 8.50 GB, leaving about 39.8 GB after both pools. On each decode rank,
the KDA/Mamba pool admits 656 states and consumes 0.79 GB of convolution state,
22.46 GB of SSM state, 33.09 GB of intermediate SSM state, and 1.16 GB of
intermediate convolution state. Decode allocates 10,005,632 target KV tokens
in 66.01 GB plus the matched draft pool in 6.00 GB, leaving about 39.0 GB before
graph capture. Full target verification graphs consume another 4.00 GB per
rank; the adaptive draft decode/extend graph tiers reduce the final reported
free memory to about 15.3 GB. These are stable allocations, not the stale
75-GB draft-model estimate used in earlier diagnosis.

## Trace attribution

The decode-side `transfer_duration` timer is not pure XGMI latency. Matching
prefill and decode `ReqTimeStats` by bootstrap room shows that it spans prefill
queueing, prefill forward execution, and the overlapped early-send transfer.

For all 215 matched r22 profiling requests:

- Prefill queue: 12.69 ms average, 1.47 ms p50, 2.43 ms p90
- Prefill forward: 2,050.59 ms average, 2,068.49 ms p50, 2,229.55 ms p90
- Decode transfer timer: 2,082.73 ms average, 2,088.15 ms p50, 2,232.85 ms p90
- Timer residual after subtracting prefill queue, forward, and bootstrap:
  -2.11 ms average, -1.77 ms p50, 45.30 ms p90
- Prefill-reported MoRI transfer speed: 90.42 GB/s average, 72.55 GB/s p50,
  179.39 GB/s p90
- Decode queue: 0.10 ms average

The attempted full C16 run was externally interrupted during cold warmup. Its
15 matched requests show the same attribution: the decode transfer timer is
prefill wait, not serialized XGMI. Prefill queue averaged 32.61 seconds and
prefill forward averaged 11.45 seconds for uncached prompts up to 433K tokens;
the matched transfer residual averaged -23.85 ms.

The currently valid prefill profile ranks the optimization targets as:

1. RCCL/CP collectives, about 29% combined
2. AITER MoE, 16.18%
3. FP8 DSA MQA logits, 13.1%
4. Copies, about 4.8%

That profile predates the current FlyDSL MQA wiring and must be repeated. The
decode profile also predates the current adaptive graph and mHC/KDA fixes; it
is diagnostic only and cannot establish the final decode ranking.

The stale r13 decode capture can now be attributed precisely. Five full target
graph replays on each TP rank spent roughly 29-30% of summed kernel time in the
FP32 add, divide, and reduction sequence implementing mHC's iterative Sinkhorn
normalization. AITER MoE was about 11.5%, the FP8 blockscale GEMMs about 7.7%,
and INT4 QuickReduce 4.0-5.7%, depending on rank. Runtime commit `bdce464a70`
replaced the pure-PyTorch mHC pre/post sequence with AITER kernels; r22 loaded
`module_mhc` on all prefill and decode ranks. A fresh decode profile must first
confirm that removal before any optimization is attributed to the old
elementwise sequence.

Speculative decode is already functioning in r22 rather than merely enabled.
Across 237 decode log samples during the canonical smoke, adaptive EAGLE
averaged 5.75 accepted tokens with p50 5.69 and an average/p50 acceptance rate
of 0.79. The adaptive controller selected 3-, 5-, and 7-step gears 32, 115, and
90 times respectively. Every sampled batch used a target CUDA graph, with no
retractions; average instantaneous server generation throughput was 257.05
tok/s across an average 1.83 active requests.

## Current candidates

Runtime SGLang candidate: `a8447ac53a8bb4a8b7d9cb203dfcf728fed8ca4c`.

- Adaptive speculative graph capture now sizes every gear and shares one
  process-wide capture stream, reducing graph scratch pressure on KV/KDA.
- The checkpoint's gfx942 FlyDSL ragged FP8 MQA resolver is wired into both the
  legacy and GLM-5.3 kpool indexers. The launcher exposes a matched on/off
  switch and defaults it on.
- The launcher now separates decode speculative arguments from prefill
  draft-state arguments. `ENABLE_PREFILL_DRAFT_STATE=0` enables the target-only
  prefill A/B without disabling EAGLE on decode; the matched path remains the
  default.
- A minimal AITER `a5b691e3` overlay contains only the GLM-5 FP8 tuned CSV and
  three gfx942 code objects. It replaces the exact old `32x256` kernel loaded by
  r22 for M=32-256. AITER's isolated tuned times improve 10.4-13.0%, while the
  new binaries also fix a missing scale barrier.

Dormant CR7 containers:

- Baseline: `glm53-pd-r24-head-a8447ac-cr7`
- AITER overlay: `glm53-pd-r25-aiter-a5b691e3-cr7`

They must be started only with exact GPUardian authorization.

## ROCm library audit

The pinned image is not carrying a recent `rocm-libraries` snapshot. Its
installed packages are ROCm 7.2.0, hipBLASLt 1.2.1, rocBLAS 5.2.0, and RCCL
2.27.7. The hipBLASLt binary reports git revision `5b515cf1bc`, while the
synced `hdt98/rocm-libraries` fork is at `ac26ef164c7`.

The r18 prefill trace makes the useful part of that delta precise. On one TP
rank, hipBLASLt/Tensile kernels account for 6.43% of summed GPU kernel time.
Within that subtotal, 99.44% is the gfx942 `BBS` family and only 0.56% is
single precision; HHS is effectively zero. The hottest `BBS` kernel alone is
4.55% of total kernel time. It is the
`Cijk_Alik_Bljk_BBS...MT256x224x64` kernel: 33,943 dispatches, 13.606 seconds
summed, and 400.8 microseconds per dispatch. This is a direct BF16/TN tuning
target rather than a generic reason to replace every ROCm math library.

Relevant upstream work is therefore limited to the gfx942 `BBS` tuning line:

- `904b35244dd`: large-K BBS grid tuning
- `d930bfea324`: exact BBS/TN tuning
- `f88ee60fb47`: broader gfx942 BBS grid and equality tuning
- `8c12dadbfd8`: later gfx942 BBS/TN grid tuning

The first matched library experiment should use an isolated hipBLASLt
1.2-series build through `f88ee60fb47`, because it retains the image's major
ABI while including the direct BBS tuning. The later `8c12dadbfd8` source is
hipBLASLt 1.4.0 and must not be overlaid as though it were ABI-equivalent.
Neither build should replace the node or image libraries globally.

The newly synced `ac26ef164c7` head is hipBLASLt 1.4.1, but none of its newest
gfx942 commits supersedes that matched experiment. `253389de6ca` retunes
single-precision GEMM, `68ce6209fcd` changes rocBLAS's single-precision backend,
and `503228a1aaf` adds F8NBS sizes; those families are absent or negligible in
the trace. `669ef4d1fca` looks relevant by name but adds MXFP4/FP8 Origami
libraries only for gfx1250. The synced head is therefore a source of candidate
logic, not a safe whole-library drop-in for the ROCm 7.2 image.

Several superficially relevant commits are excluded: the 38-CU
WorkgroupMappingXCC fix targets CPX, while CR7 is SPX/NPS1; HHS and F8NBS
tuning do not match the traced datatype family; CK's CompV4 barrier removal is
already older than the CK pins in both the baseline and candidate AITER trees;
and RCCL is not present in this monorepo. Since hipBLASLt is only 6.43% of the
current prefill trace, this A/B follows the current-head, FlyDSL, AITER, and
topology cells unless a new trace raises its share.

## Experiment order

1. Current HEAD smoke C8 with FlyDSL enabled.
2. Clean restart, then full uncapped C16.
3. Matched FlyDSL off/on smoke if the new prefill trace does not prove the gain.
4. Matched AITER baseline/`a5b691e3` smoke.
5. Matched isolated hipBLASLt 1.2 baseline/`f88ee60fb47` smoke if the new
   trace still shows material `BBS` time.
6. Role-local KDA cap: prefill max-running 16 with KDA cap 64; decode
   max-running 24 with KDA cap 96. Verify startup capacity before benchmarking.
7. Matched prefill draft-state on/off smoke with
   `ENABLE_PREFILL_DRAFT_STATE`. A GLM-5.2 Messi target-only
   prefill produced a correct 131K-token NIAH answer, so the mode is
   semantically viable, but its C4 GSP attempts completed zero warmup
   sequences and do not establish a performance win. Promote it only if
   AgentX shows lower TTFT without degrading speculative acceptance or decode
   throughput.
8. Prefill topology cells, in trace order: TP4 without CP, TP4/EP4 MoRI A2A,
   then DP2 x TP2 if the collective profile still dominates.
9. Profile the winning prefill and decode cells separately with the same seeded
   AgentX replay.
10. Run full C16 and increase concurrency until one of the four gates or
   resident KV/KDA capacity fails. Confirm the last passing point with a clean
   restart.

Every final frontier row must include ISL/OSL distribution, request counts and
errors, TTFT, per-user decode throughput, node TPS, ITL/TPOT, input/prefill
throughput, effective concurrency, cache hit, KV/KDA capacity, evictions,
retractions, transfer timing, and scheduling queue behavior.
