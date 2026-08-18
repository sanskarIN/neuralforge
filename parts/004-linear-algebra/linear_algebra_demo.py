"""Demonstrate core linear algebra operations used by neural networks."""

from __future__ import annotations

from neuralforge.linear_algebra import cosine_similarity, dot, matmul, outer, transpose


def main() -> None:
    vector_a = [1.0, 2.0, 3.0]
    vector_b = [4.0, 5.0, 6.0]
    matrix_a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    matrix_b = [[0.5, 1.0], [1.5, -1.0], [2.0, 0.25]]

    print("NeuralForge Part 004 — linear algebra")
    print(f"dot(a, b): {dot(vector_a, vector_b)}")
    print(f"cosine(a, b): {cosine_similarity(vector_a, vector_b):.6f}")
    print(f"transpose(A): {transpose(matrix_a)}")
    print(f"A @ B: {matmul(matrix_a, matrix_b)}")
    print(f"outer([1, 2], [3, 4]): {outer([1, 2], [3, 4])}")


if __name__ == "__main__":
    main()
