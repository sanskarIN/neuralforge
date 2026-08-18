# Part 004 — Linear Algebra for Neural Networks

This companion material implements core vector and matrix operations with explicit Python loops so learners can see what later tensor libraries accelerate.

## Covered concepts

- dot products;
- Euclidean norms;
- cosine similarity;
- matrix transpose;
- matrix multiplication;
- outer products;
- shape and finite-value validation.

## Run the demo

From the repository root:

```bash
PYTHONPATH=src python parts/004-linear-algebra/linear_algebra_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/004-linear-algebra/linear_algebra_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_linear_algebra -v
```

The reusable implementation lives in `src/neuralforge/linear_algebra.py`.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
