import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ReachedLayerForward(Exception):
    pass


class TestNextSwizzledInputGuard(unittest.TestCase):
    def test_none_hidden_state_reaches_layer_forward(self):
        """A deferred HIP mHC boundary may carry the state outside this value."""
        for is_blackwell in (False, True):
            with self.subTest(is_blackwell=is_blackwell):
                first_layer = mock.Mock()
                first_layer.engram = None
                first_layer.hc_boundary_fused = False
                first_layer.forward_hc_pre_from_prev.side_effect = (
                    _ReachedLayerForward
                )
                second_layer = mock.Mock()

                model = SimpleNamespace(
                    pp_group=SimpleNamespace(world_size=1),
                    engram_hasher=None,
                    engram_prefetch_stream=None,
                    late_layer_start=None,
                    start_layer=0,
                    end_layer=2,
                    layers=[first_layer, second_layer],
                    config=SimpleNamespace(model_type="deepseek_v41"),
                    dspark_layers_to_capture=None,
                )

                with (
                    mock.patch.object(deepseek_v4, "is_cp_active", return_value=False),
                    mock.patch.object(
                        deepseek_v4, "check_cuda_graph_backend", return_value=True
                    ),
                    mock.patch.object(
                        deepseek_v4,
                        "get_platform",
                        return_value=SimpleNamespace(is_blackwell=is_blackwell),
                    ),
                    self.assertRaises(_ReachedLayerForward),
                ):
                    deepseek_v4.DeepseekV4Model._forward_layers_hc_pre_from_prev(
                        model,
                        positions=object(),
                        hidden_states=None,
                        forward_batch=SimpleNamespace(),
                        input_ids=object(),
                        input_ids_global=object(),
                        capture_dspark=False,
                        dspark_aux_hidden_states=[],
                    )


if __name__ == "__main__":
    unittest.main()
