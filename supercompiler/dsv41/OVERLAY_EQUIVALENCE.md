# DSv4.1 banked overlay equivalence

## Identity

- Source image: `lmsysorg/sglang-rocm@sha256:3a75c91365cc4f4e1194bf5d64b92f2f1ef134d8b9045da165fba28a8dada464`
- Image ID: `bc9c6e337f99e492ef4e95d94b2b3474b718c0fd7071db9f8550f0346032fdc7`
- Exact SGLang HEAD inside the image: `e262b44eef1f2bf83e34345276e2bb91a086587e`
- Canonical exported manifest SHA-256: `dd44f298e3fe125750da76a0bf3a8df505aa7eb124cdc76d85830d12d1031c53`
- Canonical `sha256sums.txt` SHA-256: `00df0916325e52d964ecaeabf4b79c9b657b19db1b966e66fd6aaf68b534010f`

Both final recipes resolve to the same 26 mounts:

- `recipe_pd1p2d_c160_r38_costmodel_20261105`
- `recipe_pd1p1d_c192_pp_overlap_budget16k_r38`

## Banked runtime

Run root:

```text
/data/sonle5/dsv41_restage_20261019/pd_c64_20261021/pd1p1d_c192_r38_gamma_2367_20261105/runs/pd1p1d_c192_r38_gamma_2367_20261105_beta_unpinned_20261105T044606Z
```

Result:

```text
nexusprof_experiments/pd_c64_20261021/RESULT_pd1p1d_c192_r38_gamma_2367_20261105.md
sha256 458a835cfd1076151f3e0a098a839a1aa2de6aca6e2111befaa3684c1b841670
```

The result measured 1,813.3 plateau tok/s/GPU, 1,774.8 ms mean
TTFT, 10.292525 ms request-mean ITL, and matched B300 ratio 0.69203.

## CPU proof

`verify_overlay.py` reconstructs every runtime target from this branch,
independently hashes the retained extraction, checks each target against the
canonical manifest, and recreates the overlay set with the recorded mode.

```text
python3 supercompiler/dsv41/verify_overlay.py \
  --reference <retained-export>/sources \
  --output /tmp/OVERLAY_EQUIVALENCE.json
26/26 mounts byte-equal
```

The recorded proof is `OVERLAY_EQUIVALENCE.json`. Mount 0 is a directory and
uses the exporter's path-sorted tree hash. All 31 physical files, including
the `__pycache__/*.pyc` files, are retained. The three external mounts remain
under `supercompiler/dsv41/external/`:

- index 0: fastokens site package
- index 8: `module_mla_v4_prefill_opus.so`
- index 22: P-unit fused-MoE tuning CSV

## Attribution boundary

`LEVER_GROUPS.json` maps every import to a source-lineage commit and component
result. `stack_r2` is explicitly a composite bank: the commit split preserves
ownership and reviewability, not independent performance attribution.

`mega_mx` is not in either final recipe's 26 mounts. It is recorded as
unrepresented rather than silently included.
