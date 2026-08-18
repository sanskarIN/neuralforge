# Part 010 — Build a Perceptron and Logistic Classifier from Scratch

This companion part trains two foundational binary models without a machine-learning framework so every update is visible.

## Model 1 — Perceptron

`src/neuralforge/perceptron.py` implements:

- linear score `w·x + b`;
- hard binary decision;
- classic mistake-driven weight/bias updates;
- optional seeded example shuffling;
- early stop when an epoch makes zero mistakes;
- mistake history for inspecting convergence.

A perceptron can solve linearly separable datasets, but it does not produce calibrated probabilities and cannot solve inherently non-linearly-separable patterns such as XOR without feature transformation or additional layers.

## Model 2 — Logistic regression

`src/neuralforge/logistic_regression.py` implements:

- linear logits;
- numerically stable sigmoid probabilities;
- binary cross-entropy loss;
- full-batch gradient descent;
- optional L2 weight regularization;
- configurable probability threshold;
- loss history for checking optimization progress.

## Run the comparison demo

```bash
PYTHONPATH=src python parts/010-perceptron-logistic/classifier_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/010-perceptron-logistic/classifier_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_perceptron tests.test_logistic_regression -v
```

## Next connection

Parts 011–012 can build on these explicit forward computations by introducing computational graphs, automatic differentiation concepts, and backpropagation through multilayer networks.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
