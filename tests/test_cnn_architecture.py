from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.cnn_architecture import (  # noqa: E402
    ConvLayerSpec,
    PoolLayerSpec,
    analyze_cnn_architecture,
    conv_mac_count,
    conv_parameter_count,
    lenet_style_specs,
    residual_compatible,
)


class CNNArchitectureTests(unittest.TestCase):
    def test_standard_convolution_parameter_count(self) -> None:
        self.assertEqual(conv_parameter_count(3, 64, 3), 1792)
        self.assertEqual(conv_parameter_count(3, 64, 3, bias=False), 1728)

    def test_grouped_convolution_cost(self) -> None:
        self.assertEqual(conv_parameter_count(3, 6, 3, groups=3), 60)
        self.assertEqual(conv_mac_count(8, 8, 3, 6, 3, groups=3), 3456)
        with self.assertRaises(ValueError):
            conv_parameter_count(5, 8, 3, groups=2)

    def test_lenet_style_feature_extractor(self) -> None:
        report = analyze_cnn_architecture((32, 32, 1), lenet_style_specs())
        self.assertEqual(report.output_shape, (5, 5, 16))
        self.assertEqual(report.total_parameters, 2572)
        self.assertEqual(report.total_multiply_accumulates, 357600)
        self.assertEqual(report.receptive_field, 16)
        self.assertEqual(
            tuple(layer.output_shape for layer in report.layers),
            ((28, 28, 6), (14, 14, 6), (10, 10, 16), (5, 5, 16)),
        )

    def test_stride_and_padding_change_shape_and_receptive_field(self) -> None:
        report = analyze_cnn_architecture(
            (32, 32, 3),
            [
                ConvLayerSpec(16, 3, stride=2, padding=1, name="stem"),
                ConvLayerSpec(16, 3, padding=1, name="block"),
            ],
        )
        self.assertEqual(report.output_shape, (16, 16, 16))
        self.assertEqual(report.layers[0].receptive_field, 3)
        self.assertEqual(report.layers[0].output_jump, 2)
        self.assertEqual(report.receptive_field, 7)

    def test_residual_shape_compatibility(self) -> None:
        self.assertTrue(residual_compatible((28, 28, 64), (28, 28, 64)))
        self.assertFalse(residual_compatible((14, 14, 128), (28, 28, 64)))

    def test_pooling_has_no_trainable_parameters(self) -> None:
        report = analyze_cnn_architecture((8, 8, 4), [PoolLayerSpec(2, 2)])
        self.assertEqual(report.total_parameters, 0)
        self.assertEqual(report.total_multiply_accumulates, 0)
        self.assertEqual(report.output_shape, (4, 4, 4))

    def test_invalid_architectures_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_cnn_architecture((32, 32, 3), [])
        with self.assertRaises(ValueError):
            analyze_cnn_architecture((4, 4, 3), [ConvLayerSpec(8, 7)])


if __name__ == "__main__":
    unittest.main()
