# Part 017 — Learning-Rate Schedules and Optimization Control

Part 017 separates **how parameters are updated** from **how the learning rate changes over time**. Optimizer state lives in Part 013; this part supplies deterministic schedule functions and a stateful validation-plateau controller.

## Implemented schedules

- constant learning rate;
- step decay;
- exponential decay;
- cosine decay with a minimum floor;
- linear warmup;
- warmup followed by cosine decay;
- validation-driven `ReduceLROnPlateau` for both minimization and maximization metrics.

Reusable implementation: `src/neuralforge/schedules.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/017-learning-rate-schedules/schedule_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/017-learning-rate-schedules/schedule_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_schedules -v
```

## Control principles

- A schedule should be deterministic from its inputs.
- Warmup can reduce abrupt early updates when a model is sensitive at initialization.
- Decay schedules lower the step size as training progresses.
- Plateau control should watch a validation metric rather than repeatedly reacting to training noise.
- A learning-rate controller does not replace gradient diagnostics or good initialization.

## Reproducibility

Schedule functions are pure and do not use random state. `ReduceLROnPlateau` stores only the current rate, best metric, and bad-epoch counter, making its state easy to log in an experiment record.

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
