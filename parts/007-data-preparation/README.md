# Part 007 — Data Preparation and Leakage-Resistant Evaluation

This companion module focuses on a critical evaluation rule: information from validation/test examples must not influence state learned during training-data preparation.

## Covered concepts

- seeded train/validation/test splits;
- mutually exclusive index partitions;
- classification-aware stratified splitting;
- deterministic row selection;
- column-wise standardization;
- fitting preprocessing statistics on training rows only;
- reusing the fitted training state for validation/test transforms;
- shape, ratio, index, and finite-value validation.

The implementation lives in `src/neuralforge/data_preparation.py` and uses only the Python standard library.

## Leakage-resistant order of operations

1. Split raw examples into train/validation/test partitions.
2. Fit preprocessing state **only on training examples**.
3. Transform training data with that state.
4. Transform validation/test data with the same unchanged state.
5. Tune decisions using validation results.
6. Keep test results for the final unbiased evaluation.

Do not standardize the full dataset before splitting: that lets validation/test distribution information influence training preprocessing.

## Run the demo

```bash
PYTHONPATH=src python parts/007-data-preparation/leakage_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/007-data-preparation/leakage_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_data_preparation -v
```

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
