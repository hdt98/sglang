# Kimi K3 DCP prefix BCG device gate

Run the l13 TP1 tier on ROCm with:

1. Fresh prefill, then a 256-fresh-token, cached-prefix EXTEND under BCG with
   `SGLANG_ENABLE_K3_DCP_PREFIX_BCG_REPLAY=1`.
2. A prefix-cached multi-turn long-prompt passkey probe (at least 32k context).
3. A GSM8K passkey gate using the same cached-prefix multi-turn setup.
4. Logit parity against eager on identical prompts: prefer bitwise equality
   for BF16 logits, otherwise require KL divergence below 1e-3.

Passing means the cached-prefix passkey answer is still located, GSM8K stays
within the eager-run error band, and decode-only latency is unchanged. The
falsifier is any passkey miss, a GSM8K regression outside the eager band, or
KL divergence above 1e-3.

Sealed prediction for the l13 TP1 tier at 256 fresh tokens: cached-prefix EXTEND
wall drops from 42 ms to 18–25 ms, dominated by the already-measured device
time because BCG replay removes the eager host dispatch. Decode remains
unchanged.
