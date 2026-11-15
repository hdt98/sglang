# Kimi K3 capture-safe MHA prefix BCG device gate (l13 TP1, DCP1)

`SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY=1` enables the capture-safe MHA companion
route on HIP at `dcp_size == 1`. Both arms run
`--cuda-graph-backend-prefill=breakable`.

The treatment keeps fresh-token preparation captured, but moves the
prefix-dependent work into one `eager_on_graph` body that reads the live
`ForwardBatch`: prefix index gather, latent gather/dequant,
`kv_b_proj` up-projection, K/V assembly, prefix attention, and LSE merge.
MHA flags are set inside that break, and its graph-owned output is fresh-token
sized. The control leaves the variable unset or explicitly `0` and takes the
existing eager MHA prefix path.

Required CPU preflight, in the pinned ROCm image with fake `rocminfo` and the
`blake3` overlay, before any GPU lease:

1. Replay eligibility: cached-prefix EXTEND is admitted only with the treatment
   flag, while fresh EXTEND remains replayable with the flag off.
2. Break boundary: exactly one break is recorded; no eager body runs during
   capture.
3. No prefix-sized tensor is allocated in a captured segment; the allocated
   bridge output is fresh-token sized.
4. CPU correctness: the composite-break output matches the existing eager MHA
   prefix helper on small random tensors with the backend mocked away.

Device gate on `ait-gamma`, one MI350X VF (`0x75b0`, gfx950) per arm, two
concurrently on free GPUs, Podman only, GPUardian claim mode. Never reserve
GPUs, never use `--unsafe-override`, never touch another tenant's container,
and write the sealed prediction JSON into the run directory before any lease.

Use the existing l13 TP1 cell as the template:

- image: `sha256:3fa7e5b998e5ec8375b3b652a3380375356039321c245d8f862e99817f4e172e`
- nexusprof snapshot: `903e72edf615944be7ddfc23ea6d7a4aac57a5ba`
- model: `kimi_k3_rank0_l13_mla12_realw`
- control: variable unset or `0`
- treatment: `1`

Run and capture a torch profiler trace of the join windows:

1. Logit parity against eager/control on identical prompts: bitwise, or KL
   below `1e-3`, on a 30k-prefix EXTEND and on fresh prefill. Pruned tiers
   cannot pass task accuracy; size prompts with the tokenizer.
2. Join workload: 30k prefix, 256 and 4,096 fresh tokens, 0 and 4 residents,
   3 repeats. Record `hipGraphLaunch` counts inside EXTEND windows. Control
   must be zero; treatment must be greater than zero.
3. Decode workload in both arms after the same join setup.

Sealed prediction, l13 TP1, 256 fresh tokens with a 30k prefix: EXTEND wall
drops from the 42.4 ms eager baseline to about 28 ms; pass band is 22–33 ms.
Decode wall is unchanged within 2%. Report EXTEND wall, device busy, host gap,
fresh-prefill time at 4,096 and 8,192 tokens, and every seal band.

Falsifiers:

1. Logit mismatch outside bitwise or KL `< 1e-3` on either parity prompt.
2. Zero `hipGraphLaunch` in a treatment EXTEND window.
3. Treatment 256-token EXTEND wall above 33 ms.
4. Decode regression greater than 2%.
5. Any capture failure or capture limitation; report the verbatim error and
   mechanism. Do not patch around it.
