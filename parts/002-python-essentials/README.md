# Part 002 — Python Essentials for Tensor Programs

This companion material focuses on the Python data-structure ideas that become important when working with tensors: nested sequences, shape, element count, flattening, and reshaping.

The shared implementation in `src/neuralforge/tensor_basics.py` intentionally uses no third-party numerical library. The goal is to make tensor layout concepts visible before Part 3 introduces vectorized array tooling.

## Run the demo

From the repository root:

```bash
PYTHONPATH=src python parts/002-python-essentials/tensor_shape_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/002-python-essentials/tensor_shape_demo.py
```

## Covered concepts

- scalar, vector, matrix, and higher-rank shapes;
- rectangular versus ragged nested data;
- row-major flattening;
- element counting;
- explicit reshape validation;
- clear errors for incompatible shapes.

## Tests

```bash
PYTHONPATH=src python -m unittest tests.test_tensor_basics -v
```

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
