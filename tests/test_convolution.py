from __future__ import annotations

import unittest

from neuralforge.convolution import (
    as_matrix,
    average_pool2d,
    convolution2d,
    cross_correlate2d,
    max_pool2d,
    output_shape_2d,
    pad2d,
    spatial_output_size,
)


class ConvolutionTests(unittest.TestCase):
    def test_cross_correlation_matches_hand_computed_result(self) -> None:
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        kernel = [[1, 0], [0, -1]]
        self.assertEqual(
            cross_correlate2d(image, kernel),
            ((-4.0, -4.0), (-4.0, -4.0)),
        )

    def test_mathematical_convolution_flips_kernel(self) -> None:
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        kernel = [[1, 0], [0, -1]]
        self.assertEqual(
            convolution2d(image, kernel),
            ((4.0, 4.0), (4.0, 4.0)),
        )

    def test_stride_padding_and_dilation_shape_math(self) -> None:
        self.assertEqual(spatial_output_size(7, 3, stride=2, padding=1), 4)
        self.assertEqual(spatial_output_size(7, 3, dilation=2), 3)
        self.assertEqual(
            output_shape_2d((7, 9), (3, 5), stride=(2, 2), padding=(1, 2)),
            (4, 5),
        )

    def test_padding_is_symmetric(self) -> None:
        self.assertEqual(
            pad2d([[1, 2], [3, 4]], 1),
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 2.0, 0.0),
                (0.0, 3.0, 4.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
        )

    def test_dilated_correlation_uses_spaced_kernel_samples(self) -> None:
        image = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
        kernel = [[1, 0], [0, 1]]
        result = cross_correlate2d(image, kernel, dilation=2)
        self.assertEqual(result[0][0], 14.0)  # image[0][0] + image[2][2]
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)

    def test_max_and_average_pooling(self) -> None:
        image = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
        self.assertEqual(max_pool2d(image), ((6.0, 8.0), (14.0, 16.0)))
        self.assertEqual(average_pool2d(image), ((3.5, 5.5), (11.5, 13.5)))

    def test_overlapping_pooling_stride(self) -> None:
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(
            max_pool2d(image, kernel_size=2, stride=1),
            ((5.0, 6.0), (8.0, 9.0)),
        )

    def test_invalid_matrices_and_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            as_matrix([])
        with self.assertRaises(ValueError):
            as_matrix([[1, 2], [3]])
        with self.assertRaises(ValueError):
            spatial_output_size(2, 5)
        with self.assertRaises(ValueError):
            cross_correlate2d([[1]], [[1]], stride=0)


if __name__ == "__main__":
    unittest.main()
