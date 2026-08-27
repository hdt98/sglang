import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMhcDeepGemmGate(CustomTestCase):
    def test_falls_back_when_deep_gemm_is_unavailable(self):
        with (
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
                False,
            ),
        ):
            self.assertFalse(mhc._use_deep_gemm_hc_prenorm())

    def test_preserves_deep_gemm_path_when_available(self):
        with (
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
                True,
            ),
        ):
            self.assertTrue(mhc._use_deep_gemm_hc_prenorm())

    def test_explicit_disable_takes_precedence(self):
        with (
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
                True,
            ),
        ):
            self.assertFalse(mhc._use_deep_gemm_hc_prenorm())


class TestMhcTilelangBackendGate(CustomTestCase):
    def test_cuda_keeps_env_selected_tilelang_pre_path(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertTrue(mhc._use_tilelang_mhc_pre())

    def test_hip_always_uses_torch_pre_path(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            patch.object(mhc, "is_hip", return_value=True),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_pre())

    def test_explicit_tilelang_pre_disable_takes_precedence(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_pre())

    def test_cuda_keeps_env_selected_tilelang_post_path(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertTrue(mhc._use_tilelang_mhc_post())

    def test_hip_always_uses_torch_post_path(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
            patch.object(mhc, "is_hip", return_value=True),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_post())

    def test_explicit_tilelang_post_disable_takes_precedence(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_post())


class TestMhcAiterBackendGate(CustomTestCase):
    def test_hip_uses_aiter_when_enabled(self):
        with (
            envs.SGLANG_USE_AITER.override(True),
            patch.object(mhc, "is_hip", return_value=True),
        ):
            self.assertTrue(mhc._use_aiter_mhc())

    def test_cuda_does_not_use_aiter_mhc(self):
        with (
            envs.SGLANG_USE_AITER.override(True),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertFalse(mhc._use_aiter_mhc())

    def test_explicit_aiter_disable_takes_precedence(self):
        with (
            envs.SGLANG_USE_AITER.override(False),
            patch.object(mhc, "is_hip", return_value=True),
        ):
            self.assertFalse(mhc._use_aiter_mhc())

    def test_aiter_pre_dispatch_takes_precedence(self):
        residual = torch.zeros(2, 4, 8)
        post = torch.zeros(2, 4, 1)
        comb = torch.zeros(2, 4, 4)
        layer_input = torch.zeros(2, 8)
        with (
            patch.object(mhc, "_use_aiter_mhc", return_value=True),
            patch.object(
                mhc,
                "_mhc_pre_aiter",
                return_value=(post, comb, layer_input),
            ) as aiter_pre,
            patch.object(mhc, "_mhc_pre_torch") as torch_pre,
        ):
            result = mhc._mhc_pre_dispatch(
                residual=residual,
                fn=torch.zeros(24, 32),
                hc_scale=torch.ones(3),
                hc_base=torch.zeros(24),
                rms_eps=1e-6,
                hc_pre_eps=1e-6,
                hc_sinkhorn_eps=1e-6,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=20,
            )

        self.assertIs(result[0], post)
        self.assertIs(result[1], comb)
        self.assertIs(result[2], layer_input)
        self.assertFalse(result[3])
        aiter_pre.assert_called_once()
        torch_pre.assert_not_called()

    def test_aiter_post_dispatch_takes_precedence(self):
        x = torch.zeros(2, 8)
        residual = torch.zeros(2, 4, 8)
        post = torch.zeros(2, 4, 1)
        comb = torch.zeros(2, 4, 4)
        expected = torch.empty_like(residual)
        with (
            patch.object(mhc, "_use_aiter_mhc", return_value=True),
            patch.object(
                mhc, "_mhc_post_aiter", return_value=expected
            ) as aiter_post,
            patch.object(mhc, "_mhc_post_torch") as torch_post,
        ):
            result = mhc._mhc_post_dispatch(x, residual, post, comb)

        self.assertIs(result, expected)
        aiter_post.assert_called_once_with(x, residual, post, comb)
        torch_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
