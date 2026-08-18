# Part 017 — Learning-Rate Schedules and Optimization Control

Part 017 separates the optimizer algorithm from the rule that decides **how large each update should be over time**.

## Implemented concepts

- constant learning rate;
- step decay;
- exponential decay;
- cosine decay with a minimum learning rate;
- linear warmup;
- warmup followed by cosine decay;
- validated runtime learning-rate updates for NeuralForge optimizers;
- validation-metric `ReduceLROnPlateau` control with patience, minimum improvement, floor, and reduction count.

Reusable implementation: `src/neuralforge/schedules.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/017-learning-rate-control/schedule_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/017-learning-rate-control/schedule_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_schedules -v
```

## Schedule versus optimizer

Adam, SGD, Momentum, and RMSProp define how gradients are transformed into updates. A learning-rate schedule controls the overall update scale. The two mechanisms are related but should remain independently inspectable.

## Warmup

Warmup can reduce the size of the earliest parameter updates when representations, optimizer moments, or normalized statistics are still settling. It is not a guarantee against divergence; gradient monitoring and finite-value checks remain important.

## Plateau control

A metric-driven controller responds to validation behavior rather than only to step number. Validation metrics must remain leakage-resistant: do not tune the schedule against the final held-out test set.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
