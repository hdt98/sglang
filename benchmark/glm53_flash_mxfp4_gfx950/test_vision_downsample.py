"""CPU regression gate for the separately mounted GLM-5.3-Flash overlay.

Run with that overlay on PYTHONPATH. Extract only its downsampler constructor
so this test does not initialize distributed attention or allocate the model.
Full-model numerical and cold-image latency checks remain deployment gates.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang
from sglang.srt.layers.conv import Conv2dLayer


def model_downsample(in_channels=16, out_channels=32, merge_size=2):
    source_path = Path(sglang.__file__).parent / "srt/models/glm5_next.py"
    tree = ast.parse(source_path.read_text())
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextVisionModel"
    )
    constructor = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignment = next(
        node
        for node in constructor.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "downsample"
            for target in node.targets
        )
    )
    return eval(
        compile(ast.Expression(assignment.value), str(source_path), "eval"),
        {
            "nn": nn,
            "Conv2dLayer": Conv2dLayer,
            "vision_config": SimpleNamespace(
                hidden_size=in_channels,
                out_hidden_size=out_channels,
                spatial_merge_size=merge_size,
            ),
        },
    )


class TestVisionDownsample(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)

    def test_forward_does_not_dispatch_convolution(self):
        layer = model_downsample()
        reference = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        layer.load_state_dict(reference.state_dict(), strict=True)
        # GLM OCR's inherited forward reshapes NHWC then permutes to NCHW.
        x = torch.randn(7, 2, 2, 16).permute(0, 3, 1, 2)
        with torch.no_grad():
            expected = reference(x)
            with patch(
                "torch.nn.functional.conv2d",
                side_effect=AssertionError("vision downsample invoked convolution"),
            ):
                actual = layer(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_weights_bias_and_patch_order_match_convolution(self):
        for merge_size in (2, 3):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(merge_size=merge_size, dtype=dtype):
                    layer = model_downsample(16, 32, merge_size).to(dtype)
                    reference = nn.Conv2d(
                        16, 32, kernel_size=merge_size, stride=merge_size
                    ).to(dtype)
                    layer.load_state_dict(reference.state_dict(), strict=True)
                    self.assertEqual(set(layer.state_dict()), {"weight", "bias"})
                    for patches in (1, 3):
                        x = torch.randn(
                            7, merge_size * patches, merge_size, 16, dtype=dtype
                        ).permute(0, 3, 1, 2)
                        for tensor in (x, x.contiguous()):
                            with torch.no_grad():
                                torch.testing.assert_close(
                                    layer(tensor),
                                    reference(tensor),
                                    rtol=1e-5,
                                    atol=1e-6,
                                )


if __name__ == "__main__":
    unittest.main()
