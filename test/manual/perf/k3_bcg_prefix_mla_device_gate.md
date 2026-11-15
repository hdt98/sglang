# Kimi K3 prefix BCG device gate (DCP1)

`SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY=1` pins absorbed MLA under breakable
prefill CUDA graphs on HIP at `dcp_size == 1`, so a cached-prefix EXTEND runs
AITER `mla_prefill_fwd` (576-wide absorbed head) at each eager attention break
and the graph replays. Under DCP the extend path calls CK `flash_attn`, which
rejects head dims above 256, so DCP layouts are excluded (falsified on K3
TP2 DCP2: "CK only supports head dimension at most 256" during capture).

Run the l13 TP1 tier on ROCm, both arms with
`--cuda-graph-backend-prefill=breakable`, control with the variable unset:

1. Logit parity against eager on identical prompts, including a 30k-prefix
   EXTEND: bitwise, or KL below 1e-3. Pruned tiers cannot pass task accuracy,
   so GSM8K is not the gate there. Size any passkey prompt with the tokenizer.
2. Join workload: 30k prefix, 256 and 4,096 fresh tokens, 0 and 4 residents,
   3 repeats. Prove replay with `hipGraphLaunch` counts inside the EXTEND
   windows (control: 0).
3. Fresh no-prefix prefill at 4,096 and 8,192 tokens in both arms, since the
   pin also moves fresh prefill from MHA to absorbed MLA.

Sealed prediction, l13 TP1, 256 fresh tokens: cached-prefix EXTEND wall drops
from 42 ms (eager, 17.8 ms device) to 18–25 ms; decode unchanged within 2%;
fresh 4,096-token prefill no worse than 1.15x control. Falsifiers: any logit
mismatch beyond the band, zero graph launches in treatment EXTEND windows,
EXTEND wall above 30 ms, or fresh prefill above 1.15x.
