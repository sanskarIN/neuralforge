# Part 020 — Training-Loop Engineering & Reproducible Experiment Runners

Part 020 combines the framework-light NeuralForge components into one reproducible training workflow.

## Implemented runner features

- immutable experiment configuration;
- SHA-256 configuration fingerprint;
- SHA-256 training-data fingerprint;
- combined run fingerprint;
- deterministic model initialization and shared random seeding;
- selectable SGD, Momentum, RMSProp, or Adam optimizer;
- constant, cosine, or warmup+cosine learning-rate control;
- optional global gradient clipping;
- per-epoch gradient health capture;
- post-update training-loss recording;
- optional validation-loss recording;
- optional validation-based early stopping;
- immutable epoch records;
- final/best-loss convenience properties.

Reusable implementation: `src/neuralforge/training.py`.

The educational runner currently targets full-batch scalar-output **regression** so the orchestration remains visible. Later Parts can generalize the same engineering structure to batching, classification, checkpointing, distributed execution, mixed precision, and framework-native training stacks.

## Run the demo

```bash
PYTHONPATH=src python parts/020-training-loop/experiment_runner_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/020-training-loop/experiment_runner_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_training -v
```

## Reproducibility contract

A fingerprint proves which serialized configuration/data values were supplied to this runner; it does **not** prove universal bit-for-bit reproducibility across every operating system, Python build, framework, device, or hardware backend. Record the runtime environment when publishing measured results.

## Training-loop order

Each epoch:

1. resolve and apply the current learning rate;
2. clear previous parameter gradients;
3. build predictions and the scalar training loss;
4. run reverse-mode autodiff;
5. record gradient health and pre-clipping norm;
6. optionally clip gradients;
7. update parameters with the selected optimizer;
8. evaluate updated training/validation loss;
9. record an immutable epoch result;
10. evaluate early-stopping state when configured.

This explicit order makes state transitions reviewable and gives later production-oriented Parts a clean baseline to extend.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
