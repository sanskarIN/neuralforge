# Part 009 — From a Biological Metaphor to an Artificial Neuron

This companion implementation focuses on the mathematical artificial neuron rather than treating the biological analogy as a literal model of a real brain cell.

## Covered concepts

- input features;
- trainable weights;
- bias;
- weighted contributions;
- pre-activation value `z`;
- identity, sigmoid, tanh, ReLU, and leaky-ReLU activations;
- activation derivatives;
- an inspectable forward-pass trace.

The reusable implementation lives in `src/neuralforge/neuron.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/009-artificial-neuron/neuron_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/009-artificial-neuron/neuron_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_neuron -v
```

## Conceptual boundary

The biological-neuron analogy can help introduce weighted inputs and activation, but artificial neurons are mathematical components optimized with numerical algorithms. Their behavior should be understood from the equations and code rather than from the metaphor alone.

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
