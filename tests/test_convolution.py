from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.convolution import (  # noqa: E402
    average_pool2d,
    conv2d,
    effective_kernel_size,
    max_pool2d,
    output_size,
    pad2d,
    same_padding,
)


class ConvolutionTests(unittest.TestCase):
    def test_cross_correlation_matches_manual_result(self) -> None:
        image = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        kernel = ((1, 0), (0, -1))
        self.assertEqual(conv2d(image, kernel), ((-4.0, -4.0), (-4.0, -4.0)))

    def test_true_convolution_flips_kernel(self) -> None:
        image = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        kernel = ((1, 0), (0, -1))
        self.assertEqual(
            conv2d(image, kernel, flip_kernel=True),
            ((4.0, 4.0), (4.0, 4.0)),
        )

    def test_same_padding_preserves_shape_for_stride_one(self) -> None:
        image = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        kernel = ((0, 1, 0), (1, 1, 1), (0, 1, 0))
        result = conv2d(image, kernel, padding="same")
        self.assertEqual((len(result), len(result[0])), (3, 3))
        self.assertEqual(result[1][1], 25.0)

    def test_stride_and_dilation_geometry(self) -> None:
        self.assertEqual(effective_kernel_size(3, 2), 5)
        self.assertEqual(output_size(7, 3, stride=2, dilation=2), 2)
        self.assertEqual(same_padding(5, 3, stride=2), (1, 1))

    def test_padding_uses_requested_fill(self) -> None:
        self.assertEqual(
            pad2d(((1, 2), (3, 4)), (1, 0, 1, 0), value=-1),
            ((-1.0, -1.0, -1.0), (-1.0, 1.0, 2.0), (-1.0, 3.0, 4.0)),
        )

    def test_pooling(self) -> None:
        image = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))
        self.assertEqual(max_pool2d(image), ((6.0, 8.0), (14.0, 16.0)))
        self.assertEqual(average_pool2d(image), ((3.5, 5.5), (11.5, 13.5)))

    def test_invalid_inputs_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            conv2d(((1, 2), (3,)), ((1,),))
        with self.assertRaises(ValueError):
            conv2d(((1, 2), (3, 4)), ((1, 2, 3),), padding="valid")
        with self.assertRaises(ValueError):
            max_pool2d(((1, 2), (3, 4)), kernel_size=3)
        with self.assertRaises(ValueError):
            output_size(2, 5)


if __name__ == "__main__":
    unittest.main()
