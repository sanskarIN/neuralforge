# Part 018 — Initialization and Deep Signal Propagation

Part 018 treats initialization as a variance-control problem rather than a list of formulas to memorize.

## Implemented initializers

- zeros;
- Xavier/Glorot uniform;
- Xavier/Glorot normal;
- He/Kaiming uniform;
- He/Kaiming normal;
- LeCun normal.

Reusable implementation: `src/neuralforge/initialization.py`.

## Implemented diagnostics

- explicit fan-in/fan-out initialization plans;
- deterministic seeded matrix generation;
- population-variance measurement;
- multi-layer signal propagation through linear, tanh, or ReLU activations;
- per-layer variance profiles and final/input variance ratios.

## Run the demo

```bash
PYTHONPATH=src python parts/018-initialization/signal_propagation_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/018-initialization/signal_propagation_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_initialization -v
```

## Why zero weights are different from zero bias

Zero-initializing every weight makes neurons in the same layer begin symmetrically and, in the simple propagation demo, collapses the signal immediately. Zero biases are common; zeroing all weights is generally not a useful deep-network initialization strategy.

## Activation-aware intuition

Xavier-style scaling is commonly associated with approximately symmetric activations, while He-style scaling compensates for the signal removed by ReLU-like activations. The exact behavior still depends on architecture, data distribution, normalization, residual paths, and training dynamics, so measure rather than assume.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
