# Part 020 — Training-Loop Engineering & Reproducible Experiment Runners

Part 020 combines the framework-light components from earlier parts into a structured, inspectable training system. Instead of scattering experiment state across print statements and notebook cells, the companion runner records configuration, optimization state, gradient diagnostics, and final model outputs in one reproducible result.

## Implemented features

- immutable `ExperimentConfig`;
- deterministic SHA-256-based configuration fingerprint;
- optimizer selection for SGD, Momentum, RMSProp, and Adam;
- injectable learning-rate schedules;
- finite-loss and finite-gradient guards;
- optional global gradient clipping;
- epoch-level learning-rate, loss, gradient-norm, max-gradient, and gradient/parameter-ratio records;
- deterministic small-MLP regression experiments;
- final prediction and parameter capture;
- stable JSON experiment export.

Reusable implementation: `src/neuralforge/training.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/020-training-loop-engineering/experiment_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/020-training-loop-engineering/experiment_demo.py
```

The demo writes a JSON run record under `artifacts/`. Generated experiment files should normally remain untracked unless they are intentionally selected as reference artifacts.

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_training -v
```

## Reproducibility boundary

A configuration fingerprint proves that the recorded configuration fields match; it is not a universal guarantee that every external runtime, hardware backend, library version, or nondeterministic kernel would reproduce bit-identical outputs. The current companion runner is deliberately dependency-free and deterministic so the mechanics are visible before later framework/device-specific reproducibility work.

## Training-loop order

A full-batch epoch follows this sequence:

1. apply the scheduled learning rate;
2. clear old gradients;
3. build predictions and loss;
4. reject a non-finite loss;
5. run reverse-mode autodiff;
6. reject non-finite gradients;
7. record gradient diagnostics;
8. optionally clip the global gradient norm;
9. update parameters with the optimizer;
10. append an immutable epoch record.

## Why this matters

The difference between a demo and a maintainable experiment is often operational discipline: deterministic configuration, explicit metrics, versioned code, stable outputs, and enough metadata to understand what actually ran. This part establishes that discipline before the curriculum moves into convolutional architectures.

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
