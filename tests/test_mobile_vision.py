from __future__ import annotations

import unittest

from neuralforge.mobile_vision import (
    depthwise_separable_cost,
    efficiency_ratio,
    inverted_residual_cost,
    make_divisible,
    scale_channels,
    scale_resolution,
    standard_conv_cost,
)


class MobileVisionTests(unittest.TestCase):
    def test_standard_convolution_cost(self) -> None:
        cost = standard_conv_cost((32, 112, 112), 64, kernel_size=3, padding=1)
        self.assertEqual(cost.output_shape, (64, 112, 112))
        self.assertEqual(cost.parameters, 18_432)
        self.assertEqual(cost.macs, 231_211_008)
        self.assertEqual(cost.parameter_bytes_fp32, 73_728)

    def test_depthwise_separable_convolution_reduces_cost(self) -> None:
        standard = standard_conv_cost((32, 112, 112), 64)
        mobile = depthwise_separable_cost((32, 112, 112), 64)
        self.assertEqual(mobile.parameters, 2_336)
        self.assertEqual(mobile.macs, 29_302_784)
        ratios = efficiency_ratio(standard, mobile)
        self.assertAlmostEqual(ratios["parameter_ratio"], 2_336 / 18_432)
        self.assertGreater(ratios["parameter_reduction"], 0.87)
        self.assertGreater(ratios["mac_reduction"], 0.87)

    def test_make_divisible_and_width_scaling(self) -> None:
        self.assertEqual(make_divisible(24.0), 24)
        self.assertEqual(make_divisible(17.0), 16)
        self.assertEqual(scale_channels(32, 0.75), 24)
        self.assertEqual(scale_channels(16, 0.35), 8)

    def test_resolution_scaling(self) -> None:
        self.assertEqual(scale_resolution(224, 224, 0.75), (168, 168))
        self.assertEqual(scale_resolution(7, 5, 0.1), (1, 1))

    def test_inverted_residual_cost_and_skip_rule(self) -> None:
        block = inverted_residual_cost((24, 56, 56), 24, expansion=6, stride=1)
        self.assertEqual(block.expansion_channels, 144)
        self.assertEqual(block.expanded_shape, (144, 56, 56))
        self.assertEqual(block.output_shape, (24, 56, 56))
        self.assertEqual(block.parameters, 8_208)
        self.assertTrue(block.residual_connection)

        downsample = inverted_residual_cost((24, 56, 56), 32, expansion=6, stride=2)
        self.assertEqual(downsample.output_shape, (32, 28, 28))
        self.assertFalse(downsample.residual_connection)

    def test_biases_are_optional_and_counted(self) -> None:
        no_bias = depthwise_separable_cost((8, 16, 16), 12, bias=False)
        with_bias = depthwise_separable_cost((8, 16, 16), 12, bias=True)
        self.assertEqual(with_bias.parameters - no_bias.parameters, 20)
        self.assertEqual(with_bias.macs, no_bias.macs)

    def test_invalid_scaling_and_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scale_channels(32, 0.0)
        with self.assertRaises(ValueError):
            standard_conv_cost((0, 8, 8), 16)
        with self.assertRaises(ValueError):
            make_divisible(16, divisor=0)
        with self.assertRaises(ValueError):
            inverted_residual_cost((8, 8, 8), 8, expansion=-1)


if __name__ == "__main__":
    unittest.main()
