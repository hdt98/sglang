"""Run AITER's FMoE tuner with the isolated single-GPU recovery shim."""

import runpy

import aiter.utility.mp_tuner as _mp_tuner

from single_gpu_tuner_shim import work_group


_mp_tuner.work_group = work_group


if __name__ == "__main__":
    runpy.run_path(
        "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",
        run_name="__main__",
    )
