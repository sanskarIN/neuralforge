# Part 005 — Calculus for Neural Networks

This companion material makes derivatives and gradients executable before later parts introduce automatic differentiation and backpropagation.

## Covered concepts

- central finite-difference derivatives;
- numerical gradients for multi-dimensional objectives;
- analytical-versus-numerical gradient checking;
- absolute and relative tolerances;
- validation of non-finite inputs and invalid step sizes.

## Run the demo

From the repository root:

```bash
PYTHONPATH=src python parts/005-calculus/gradient_check_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/005-calculus/gradient_check_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_calculus -v
```

The reusable implementation lives in `src/neuralforge/calculus.py`.
