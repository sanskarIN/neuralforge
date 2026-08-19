from __future__ import annotations

import unittest

from neuralforge.cnn_architecture import ArchitectureBuilder, alexnet, lenet5, resnet18, vgg11


class CnnArchitectureTests(unittest.TestCase):
    def test_lenet5_shape_and_parameter_count(self) -> None:
        summary = lenet5(classes=10)
        self.assertEqual(summary.input_shape, (1, 32, 32))
        self.assertEqual(summary.output_shape, (10,))
        self.assertEqual(summary.total_parameters, 61_706)
        self.assertEqual(summary.layers[0].output_shape, (6, 28, 28))
        self.assertEqual(summary.layers[3].output_shape, (16, 5, 5))

    def test_alexnet_shape_flow_reaches_classifier(self) -> None:
        summary = alexnet(classes=1000)
        self.assertEqual(summary.layers[0].output_shape, (64, 55, 55))
        pool5 = next(layer for layer in summary.layers if layer.name == "pool5")
        self.assertEqual(pool5.output_shape, (256, 6, 6))
        self.assertEqual(summary.output_shape, (1000,))
        self.assertGreater(summary.total_parameters, 50_000_000)

    def test_vgg11_reduces_224_to_7_before_dense_stack(self) -> None:
        summary = vgg11(classes=100)
        pool5 = next(layer for layer in summary.layers if layer.name == "pool5")
        self.assertEqual(pool5.output_shape, (512, 7, 7))
        flatten = next(layer for layer in summary.layers if layer.operation == "flatten")
        self.assertEqual(flatten.output_shape, (25_088,))
        self.assertEqual(summary.output_shape, (100,))

    def test_resnet18_stage_transitions_and_projection(self) -> None:
        summary = resnet18(classes=10)
        stage1 = next(layer for layer in summary.layers if layer.name == "stage1_block1")
        stage2 = next(layer for layer in summary.layers if layer.name == "stage2_block1")
        stage4 = next(layer for layer in summary.layers if layer.name == "stage4_block1")
        self.assertEqual(stage1.output_shape, (64, 56, 56))
        self.assertIn("projection=no", stage1.notes)
        self.assertEqual(stage2.output_shape, (128, 28, 28))
        self.assertIn("projection=yes", stage2.notes)
        self.assertEqual(stage2.parameters, 230_144)
        self.assertEqual(stage4.output_shape, (512, 7, 7))
        self.assertEqual(summary.output_shape, (10,))

    def test_grouped_convolution_parameter_count(self) -> None:
        summary = (
            ArchitectureBuilder("grouped", (8, 16, 16))
            .conv(12, 3, padding=1, groups=4, bias=False)
            .build()
        )
        self.assertEqual(summary.output_shape, (12, 16, 16))
        self.assertEqual(summary.total_parameters, 12 * 2 * 3 * 3)

    def test_flatten_and_dense_require_compatible_shapes(self) -> None:
        builder = ArchitectureBuilder("invalid", (3, 8, 8))
        with self.assertRaises(ValueError):
            builder.dense(10)
        builder.flatten().dense(10)
        with self.assertRaises(ValueError):
            builder.conv(4, 3)

    def test_invalid_grouping_and_residual_stride_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArchitectureBuilder("groups", (3, 8, 8)).conv(8, 3, groups=2)
        with self.assertRaises(ValueError):
            ArchitectureBuilder("residual", (16, 8, 8)).residual_basic(16, stride=0)


if __name__ == "__main__":
    unittest.main()
