# Part 012 — Backpropagation Through Multilayer Perceptrons

Part 012 builds a small fully connected neural-network stack directly on the Part 011 scalar autodiff engine. Every weight and bias is a `Value`, so the same computational graph and chain-rule machinery performs backpropagation through multiple layers.

## Covered concepts

- trainable `Neuron`, `Layer`, and `MLP` modules;
- deterministic seeded initialization;
- hidden and output activations;
- parameter discovery and gradient clearing;
- mean-squared error;
- binary cross-entropy for probability outputs;
- full forward graph construction;
- backward propagation through all layers;
- a simple educational SGD update;
- tiny end-to-end regression training.

The reusable implementation lives in `src/neuralforge/nn.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/012-backpropagation/mlp_training_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/012-backpropagation/mlp_training_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_nn -v
```

## Backpropagation connection

For each batch, the demo performs the same conceptual sequence used by larger frameworks:

1. run a forward pass;
2. build a scalar loss;
3. call `loss.backward()`;
4. read gradients stored on parameters;
5. update parameter values;
6. build a fresh graph for the next step.

Part 013 replaces the basic update step with reusable optimizer classes and optimizer state.

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
