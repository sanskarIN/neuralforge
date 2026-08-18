# Part 001 — Foundations, Installation, Tooling, and Your First Neural Networks

This directory contains open-source companion material for Part 1 of NeuralForge. It is intentionally small and dependency-free so learners can inspect the mathematics and training loop before moving to array libraries and deep-learning frameworks.

## Included companion concepts

- weighted sums and bias;
- numerically stable sigmoid activation;
- binary cross-entropy;
- full-batch gradient descent;
- a single logistic neuron;
- a tiny OR-gate learning example;
- validation and reproducibility habits.

## Run the example

From the repository root:

```bash
PYTHONPATH=src python parts/001-foundations/train_or_gate.py
```

On PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/001-foundations/train_or_gate.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_foundations -v
```

The implementation shared by this lab lives in `src/neuralforge/foundations.py` so later parts can reuse and extend it without duplicating logic.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
