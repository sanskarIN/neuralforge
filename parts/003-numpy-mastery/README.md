# Part 003 — NumPy Mastery for Vectorized Computation

This companion lab moves from plain-Python tensor concepts to vectorized NumPy operations used throughout deep-learning systems.

## Environment

Part 3 uses its own pinned dependency so the rest of the repository remains lightweight:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r parts/003-numpy-mastery/requirements.txt
```

The pinned NumPy 2.5.x line requires Python 3.12 or newer. Other dependency-free repository examples can still run on the broader core Python range.

## Covered concepts

- conversion to dense numeric arrays;
- shape and dtype validation;
- column-wise standardization;
- broadcasting;
- matrix multiplication for dense layers;
- stable row-wise softmax;
- vectorized validation and tests.

## Run the demo

```bash
python parts/003-numpy-mastery/vectorization_demo.py
```

## Run Part 3 tests

```bash
python -m unittest discover -s parts/003-numpy-mastery -p "test_*.py" -v
```

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
