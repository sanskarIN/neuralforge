# Part 018 — Initialization and Deep Signal Propagation

Part 018 turns initialization formulas into executable, reproducible experiments. It provides common dense-layer initialization schemes and a forward-signal simulator that summarizes how activation distributions evolve through multiple random layers.

## Implemented initialization schemes

- zero initialization (for demonstration/testing, not a good symmetric hidden-layer default);
- fixed uniform initialization;
- Xavier/Glorot uniform;
- Xavier/Glorot normal;
- He/Kaiming uniform;
- He/Kaiming normal;
- LeCun normal;
- activation-aware default recommendations.

Reusable implementation: `src/neuralforge/initialization.py`.

## Signal-propagation report

`propagate_signal(...)` tracks, for the input and every simulated layer:

- width;
- mean;
- population variance;
- minimum and maximum activation;
- exact-zero fraction.

This makes it possible to compare initialization/activation choices before a training loop is involved.

## Run the demo

```bash
PYTHONPATH=src python parts/018-initialization-signal-propagation/signal_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/018-initialization-signal-propagation/signal_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_initialization -v
```

## Practical rule

Initialization is not about producing a particular magic distribution. The goal is to begin optimization with useful signal scales, broken parameter symmetry, and finite activations/gradients. The right choice also depends on architecture, activation, normalization, residual paths, precision, and optimizer behavior.

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
