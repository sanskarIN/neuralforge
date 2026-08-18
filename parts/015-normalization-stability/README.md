# Part 015 — Normalization and Training Stabilization in Deep Neural Networks

Part 015 adds normalization and numerically stable probability operations directly on the scalar autodiff engine.

## Implemented techniques

- differentiable batch normalization across examples;
- differentiable layer normalization within one example;
- trainable affine `gamma` and `beta` parameters;
- epsilon-based denominator stabilization;
- exponential-moving running means/variances for evaluation;
- stable softmax using a maximum-logit shift;
- gradient checks through normalization/softmax via the normal unit-test suite.

Implementation: `src/neuralforge/normalization.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/015-normalization-stability/normalization_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/015-normalization-stability/normalization_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_normalization -v
```

## Batch normalization versus layer normalization

- **Batch normalization** computes one mean/variance per feature across a batch. Training statistics and evaluation-time running statistics are therefore distinct.
- **Layer normalization** computes statistics across the features of each individual example and does not require cross-example running statistics.

## Stable softmax

Exponentiating very large logits directly can overflow. Subtracting the maximum logit first leaves softmax probabilities unchanged while keeping exponentials in a safer numeric range.

## Stabilization is layered

Normalization is only one training-stability tool. Good initialization, suitable learning rates, gradient monitoring/clipping, finite-value checks, and correctly scaled inputs remain important. Later NeuralForge parts build these ideas into larger training systems.

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
