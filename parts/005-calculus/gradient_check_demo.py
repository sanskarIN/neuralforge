"""Compare an analytical gradient with a finite-difference estimate."""

from __future__ import annotations

from neuralforge.calculus import check_gradient


def objective(point):
    x, y = point
    return x * x + x * y + 2.0 * y * y


def analytical_gradient(point):
    x, y = point
    return [2.0 * x + y, x + 4.0 * y]


def main() -> None:
    point = [1.25, -0.75]
    result = check_gradient(objective, analytical_gradient, point)

    print("NeuralForge Part 005 — gradient checking")
    print(f"point: {point}")
    print(f"analytical: {result.analytical}")
    print(f"numerical: {result.numerical}")
    print(f"absolute errors: {result.absolute_errors}")
    print(f"max error: {result.max_absolute_error:.10f}")
    print(f"passed: {result.passed}")


if __name__ == "__main__":
    main()
