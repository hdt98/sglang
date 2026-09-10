# GLM-5.3 Flash MI355 PD development checkpoint

Updated 2026-09-10. This is a shared **development/WIP checkpoint**, not a new
correctness-qualified performance release. Read this before launching this tree.

## Working agreement

- Repository: `hdt98/sglang`; push remote: `origin`.
- Branch: `users/hdt98/glm53-flash-mi355-pd-mori-staging`.
- Authoritative local worktree: `/Users/sonle5/.codex/worktrees/9f9a/sglang`.
  Do not continue in `9f9a-clean` or silently switch branches.
- Target: `OneNexus/GLM-5.3-Flash-MXFP4`, PD on **8 physical MI355X GPUs** on
  `onenexus-alpha`. The older artifact directory says MI350; Alpha is MI355X.
- Keep Mori and fixed EAGLE **5/1/6**, not adaptive speculation. Record AL/AR,
  correctness, graph replay/fallback, and request errors with every candidate.
- Use the `sglang-agentx-benchmark` skill. The comparison workload here is the
  **256K-capped AgentX corpus**, context **262,144**, **900-second scored rung**.
  Keep the skill's uncapped canonical run separate from this comparison.
- Count all 8 PD GPUs versus all 4 reference GPUs. Do not divide per-user decode
  speed by GPU count or treat normalized throughput as measured 4-GPU PD speed.
- No new GPU workload or server teardown was performed for this checkpoint.
  Check live ownership before doing either; do not stop another task's server.

## Why there were 30 uncommitted files

The pending work was one unfinished R435 experiment spread across **25 runtime
files and 5 test files**: same-worker role routing, separate scheduler queues,
local state/metadata handoff, Mori retry/diagnostics, and page-envelope copies.
Two files were untracked: the copy kernel and hybrid-PD unit tests. These were
not 30 independently validated optimizations and were not the source snapshot
that produced the best measured frontier.

This checkpoint preserves that work instead of deleting it. It also repairs a
regression found during the checkpoint review: dedicated DECODE requests with
`disagg_role=None` must still release their queued KV and hold rebootstrap during
a retract-mode pause. New tests failed before that fix and pass afterward.
The gateway's role-injection ownership assertions and decode test fixtures were
updated; duplicate imports and small lint issues were cleaned up.

### Experimental code map and limits

| Area | Main files | Status / remaining work |
|---|---|---|
| Role selection | `arg_groups/pd_disaggregation_hook.py`, OpenAI entrypoints, request schemas, gateway `pd_router.rs` | Opt-in `hybrid` worker mode and router-injected `disagg_role`; the gateway change also affects ordinary PD requests. |
| Scheduler | `managers/scheduler.py`, `disaggregation/prefill.py`, decode mixins | Separate prefill/decode state and sequential decode-first loop; not a proven elastic 8-GPU stage-allocation policy. |
| Local handoff | `disaggregation/prefill.py`, `utils.py`, `mori/conn.py` | Local KV/Mamba/metadata copy and paired-request cleanup; no qualified AgentX result for R435. |
| Memory moves | `kernels/ops/kvcache/copy_pages.py`, `mem_cache/{memory_pool,unified_memory_pool}.py` | Whole-page copy experiment also changes non-hybrid callers. GPU validation is outstanding. |
| Cleanup / tests | decode HiCache cleanup and unit tests | CPU checks do not validate GPU copy integrity, long-context correctness, or throughput. |

Specific follow-up risks, not established root causes:

1. The hybrid loop calls both planners before choosing decode over prefill.
   Prefill planning is stateful; test simultaneous runnable batches for dropped
   plans, queue ownership, and progress before promoting this scheduler.
2. Page copies require compatible page alignment and non-overlapping source and
   destination sets. Audit partial-token callers and aliasing; mocked CPU tests
   do not establish these invariants on gfx950.
3. Recheck long-context, abort/retract/resume, shared-cache lock lifetime, and
   fixed-EAGLE metadata continuity under concurrent traffic. R435 is not a
   substitute for the R425/R432 control below.

## Measured controls, not this WIP tree

**R425 C64** remains the measured output-throughput head: **4,746.98 output
tok/s**, **593.37 tok/s/GPU**, P90 per-request decode speed **189.18 tok/s/user**,
mean TTFT **5.890 s**. R425 passed pre/post GSM8K (19/20 and 20/20) and both
6-case long-context gates, to actual prompt length 216,034. This does not prove
correctness for every request up to 262,144 tokens.

Recipe: 4P+4D, TP4/EP4 per role, same-node Mori staging/XGMI, EAGLE 5/1/6,
32,768-token prefill chunks, INT4 QuickReduce, full decode graph sizes through
96; static memory fractions P=0.80 / D=0.85; strict thinking with a 2,048-token
thinking budget (4,096 total for the correctness gate). Graph capture range
alone does not prove every batch replays a graph.

**R432 C8** is the low-load control from the same 32K/INT4 recipe family:
1,301.67 output tok/s, 162.71 tok/s/GPU, P90 speed 250.99 tok/s/user.

Best observed envelope from the 2026-09-10 audit (not one fixed-recipe sweep):

| Rung | PD run | PD output tok/s | PD tok/s/GPU (8) | B300 tok/s/GPU (4) | Latest monolithic tok/s/GPU (4) |
|---|---|---:|---:|---:|---:|
| C8 | R432 | 1,301.67 | 162.71 | 241.00 | 289.94 |
| C16 | R377 | 2,139.46 | 267.43 | 361.50 | 434.90 |
| C32 | R430 | 3,349.96 | 418.75 | 420.00 | 511.14 |
| C48 | R401 | 3,991.08 | 498.88 | 430.25 | 538.70 |
| C64 | R425 | 4,746.98 | 593.37 | 320.00 | not audited |

R377 uses 16K/INT8 and R401 uses 32K/INT8; the other listed PD runs use 32K/INT4.
R377 repeats were around 2,062-2,065 TPS, so its peak is not a proven repeatable
uplift. No qualifying current-era C1/C2/C4 PD point was established. B300 C64
is dominated by its own C48 point. The newest monolithic target is a different
mixed-precision Quark L3/5/6 checkpoint on 4xMI350, not PD; its complete quality
gate chain was not re-audited here. Do not present this table as algorithm,
checkpoint, latency, or cost equivalence.

The named corpus/context/rung/EAGLE contract matches the B300 envelope. Exact
byte-identical corpus, seed/warmup/drain configuration is not fully proven by
that envelope alone. PD uses the saved AIPerf snapshot named `475d76cc`, whose
exports report 0.8.0. Its `--warmup-request-count 10` is **not** ten requests per
lane in the AgentX replay strategy. Separate warmup empty one-token responses,
scored errors, and end-of-phase cancellations; exported throughput includes the
harness's own reduction and is not simply tokens / 900.

## Trace-grounded next steps

- R423/R426 are **C64** bounded captures. R426 summaries were verified against
  single TP0 files, not four-rank aggregates. QuickReduce and mHC are measured
  prefill leads; map generic MoE/GEMM names to callsites before choosing kernels.
- Do not reuse the previous claim that BF16 copies routinely cost 3.921 seconds:
  two anomalous events account for 92.6% of that total, with thousands of same-
  stream kernels inside their reported intervals. Validate timestamps/stream
  attribution. Preserve raw events rather than silently deleting outliers.
- R434's C8 capture began in the scored phase but later hit real 300-second KV
  transfer timeouts. The first failure preceded prefill export. It is not a
  clean normal-C8 bottleneck trace. No clean C16/C32 GPU captures were established.
- Scored scheduler telemetry (nonexclusive wall-clock means): C8 prefill queue /
  forward / transfer-completion = 11.8 / 267.3 / 68.0 ms; C32 = 331.8 / 465.1 /
  252.6 ms; C64 = 2,946.6 / 1,080.2 / 720.3 ms. Do not call these pure GPU compute
  or wire-transfer time, or sum them as one request's critical path.
- R429's persistent metadata allocation pool did not demonstrate an end-to-end
  gain. Summed CPU/API/kernel durations overlap and cannot establish utilization
  or prove decode is not compute-bound.
- Once all 8 GPUs are available: reproduce the pinned R432 control unprofiled,
  run its gates, then capture **one role at a time** over a much shorter scored
  window at C8, followed by C16/C32. Correlate prefill completion, Mori submit,
  copy completion, decode admission, graph fallback and AL/AR. Promote only
  unprofiled, protocol-matched, gated runs; repeat close wins.

## Evidence and reproduction boundaries

Remote host: `onenexus-alpha`. Artifact root (historical name):
`/data/sonle5/glm53-pd-mi350-alpha-20260903`.

- Prepared control snapshot: `analysis/r436-r432-pinned-source` under that root.
  Source/tools are read-only. All **13 historical R432 recorded hashes** match;
  R432's recorded list matches R425's. Full payload manifest SHA256:
  `8067a00be227586d2477ae489fb95f44c82432432b5b87cdad2f5bd274244db4`.
  This freezes the reconstruction going forward, not proof of full historical
  identity for files absent from the old hash list. GPU revalidation is pending.
- Server image: `glm53-flash-mi355-dev:pr35546-hipfallback-20260904`, ID
  `sha256:577d1a3dfd991957b56bec7955176a8be03bab406aae9d4201570a967fc9c698`,
  **plus saved source/dependency overlays**, not a validated stock image.
- Saved control launcher: `runs/r432-c8-quickreduce-int4-run.sh`; AgentX wrapper:
  `run_r233_agentx_alpha.sh`. Old launchers mutate the shared
  `validation/r200-clean-source`; do not run them blindly. Use a separate pinned
  deployment and preserve checksum guards.
- R435 runtime: `analysis/r435-hybrid-prototype`; it may contain overlays not
  identical to this Git checkpoint. Verify hashes before copying either way.
- Run exports: `runs/<run>-agentx/profile_export_aiperf.json`, JSONL, client logs;
  sibling `runs/<run>/` contains gates, server info, and sampled metrics.
- R426 raw traces: `runs/r426-c64-r425-profile/profile/{prefill,decode}/*.trace.json.gz`.
- Detailed local audit:
  `/Users/sonle5/.codex/worktrees/9f9a-artifacts/sglang/.codex-tmp/frontier/reaudit-20260910.md`.
- B300 source supplied by user:
  `/Users/sonle5/Downloads/glm53-flash-agentx-envelope.md`.
- Monolithic exports on Gamma:
  `/data/sonle5/frontier-mixed-chunk-20260905/artifacts/b300_pinned_c{C}_realacc_agentx256k_900_20260905_hybrid_quark_l356_20260910b/profile_export_aiperf.json`.

As of the last recorded GPU check (2026-09-10 08:25:39 UTC), the unrelated
`glm53-flash-quark-mxfp4-alpha0-3-20260910T002305Z-device-isolated` occupied
Alpha 0-3; our `r435-hybrid-tp4` occupied 4-7. This is a timestamped observation,
not a reservation or a current availability guarantee. Do not message the
monolithic task `01a05180-966a-7882-8345-24d4578293b1`; inspection is read-only.

## Checkpoint verification

On macOS / Python 3.14.5, using `PYTHONPATH=python`, the focused suite passed
**62 tests and 7 subtests**, with **3 GPU tests skipped**:

```sh
PYTHONPATH=python python -m pytest -q -p no:cacheprovider \
  test/registered/unit/disaggregation/test_hybrid_pd.py \
  test/registered/unit/disaggregation/test_decode_queue_cleanup.py \
  test/registered/unit/managers/test_priority_scheduling_disaggregation.py \
  test/registered/unit/mem_cache/test_decode_radix_lock_ref.py \
  test/registered/unit/mem_cache/test_dsa_pool_host_unit.py
```

An additional 62 tests and 12 subtests passed in the disaggregation unit modules
`test_disaggregation_wire`, `test_hybrid_mla_rank_mapping`,
`test_unified_memory_move_gate`, `test_staging_draft_kv_slots`,
`test_decode_hicache_tree_core`, and `test_deferred_decode_kv_release`.
Total: **124 tests and 19 subtests passed; 3 GPU tests skipped**.

The local test environment needed the already-declared `setproctitle` dependency;
it was installed in a disposable external venv, not added to project dependencies.
Rust formatting checks passed for both touched Rust files. Gateway unit-test
execution was blocked before compilation by uncached `openai-harmony` in the
offline dependency check; no Rust compile/test success is claimed. No GPU
correctness gate or new performance result is claimed for this checkpoint.

## Git history / clean-worktree meaning

The old tracking ref was `upstream/pr/36507-latest` and Git marked it `[gone]`.
That does **not** mean GitHub PR #36507 was deleted. Tracking should be the
requested branch on `origin` (`hdt98/sglang`).

The published branch tip `de5eb5c406` had two commits replayed locally as
`ae538c983a` and `b0c82e8e91`. Added/deleted patch lines match; the first patch's
different patch-id comes from intervening context, including guarded IOEngine
initialization. Preserve both histories without reverting current source or
force-pushing. The pre-checkpoint local tip was `cdf6d507d7`.

A clean worktree means pending files were committed, **not** that this is a small
PR against latest `main`. At audit, `origin/main` was `fa6e657b93` and the local
tip was 286 commits behind / 53 ahead, with substantial upstream/refactor drift.
Do not rebase this shared branch or claim all tree differences are our fixes.
An upstream-ready extraction is separate work; keep this checkpoint recoverable.
