# Part 014 — Regularization and Generalization in Deep Neural Networks

Part 014 adds tools that influence model complexity and training duration rather than merely changing the optimizer.

## Implemented techniques

- differentiable L1 parameter penalty;
- differentiable L2 parameter penalty;
- combined regularized loss;
- deterministic-seeded inverted dropout;
- training/evaluation dropout behavior;
- early stopping with patience and minimum improvement;
- training-versus-validation generalization gap;
- parameter L2 norm monitoring.

Implementation: `src/neuralforge/regularization.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/014-regularization/regularization_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/014-regularization/regularization_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_regularization -v
```

## L1 versus L2

- **L1** adds a penalty proportional to absolute parameter magnitude and can encourage sparse solutions.
- **L2** penalizes squared magnitude and tends to shrink large parameters smoothly.

The educational L1 implementation uses `relu(x) + relu(-x)` so it remains in the same autodiff graph. At exactly zero, the chosen subgradient is zero.

## Dropout

During training, activations are independently dropped with probability `p`; kept values are scaled by `1/(1-p)`. During evaluation, dropout is disabled and activations pass through unchanged.

## Early stopping

Early stopping uses validation behavior, not training loss alone. The helper signals when validation loss has failed to improve by `min_delta` for `patience` consecutive updates.

Regularization is not a substitute for a leakage-free split: validation/test information must still remain outside training-data fitting and preprocessing state.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
