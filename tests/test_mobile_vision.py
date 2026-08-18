from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.mobile_vision import (  # noqa: E402
    depthwise_separable_cost,
    inverted_residual_cost,
    model_size_bytes,
    model_size_mebibytes,
    standard_conv_cost,
    width_scaled_channels,
)


class MobileVisionTests(unittest.TestCase):
    def test_standard_and_depthwise_separable_costs(self) -> None:
        baseline = standard_conv_cost((32, 32, 32), 64, kernel_size=3)
        mobile = depthwise_separable_cost((32, 32, 32), 64, kernel_size=3)

        self.assertEqual(baseline.parameters, 18432)
        self.assertEqual(baseline.multiply_accumulates, 18874368)
        self.assertEqual(mobile.parameters, 2336)
        self.assertEqual(mobile.multiply_accumulates, 2392064)
        self.assertEqual(mobile.output_shape, (32, 32, 64))
        self.assertLess(mobile.parameter_ratio(baseline), 0.13)
        self.assertLess(mobile.mac_ratio(baseline), 0.13)

    def test_stride_changes_mobile_output_geometry(self) -> None:
        mobile = depthwise_separable_cost((31, 31, 16), 24, stride=2)
        self.assertEqual(mobile.output_shape, (16, 16, 24))

    def test_inverted_residual_cost(self) -> None:
        block = inverted_residual_cost((16, 16, 16), 24, expand_ratio=6, stride=2)
        self.assertEqual(block.parameters, 4704)
        self.assertEqual(block.multiply_accumulates, 595968)
        self.assertEqual(block.output_shape, (8, 8, 24))

    def test_model_size_estimates_precision_effect(self) -> None:
        self.assertEqual(model_size_bytes(1_000_000, bits_per_parameter=8), 1_000_000)
        self.assertEqual(model_size_bytes(1_000_000, bits_per_parameter=32), 4_000_000)
        self.assertAlmostEqual(model_size_mebibytes(262_144, bits_per_parameter=32), 1.0)

    def test_width_scaling_rounds_to_hardware_friendly_divisor(self) -> None:
        self.assertEqual(width_scaled_channels(32, 1.4), 48)
        self.assertEqual(width_scaled_channels(16, 0.35), 8)
        self.assertEqual(width_scaled_channels(24, 1.0), 24)

    def test_invalid_cost_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            standard_conv_cost((32, 32, 0), 64)
        with self.assertRaises(ValueError):
            inverted_residual_cost((16, 16, 16), 24, expand_ratio=0)
        with self.assertRaises(ValueError):
            model_size_bytes(0)
        with self.assertRaises(ValueError):
            width_scaled_channels(32, -1.0)


if __name__ == "__main__":
    unittest.main()
