# Part 013 — Optimization Algorithms for Deep Neural Networks

Part 013 separates gradient computation from parameter-update policy. The Part 011/012 graph computes gradients; optimizer objects decide how those gradients change parameters over time.

## Implemented optimizers

- **SGD** — direct gradient descent with optional weight decay.
- **Momentum** — exponentially accumulated update direction.
- **RMSProp** — running average of squared gradients for adaptive scaling.
- **Adam** — first/second moments with bias correction.

The module also provides global L2 **gradient-norm clipping** and shared gradient clearing.

Implementation: `src/neuralforge/optim.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/013-optimization/optimizer_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/013-optimization/optimizer_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_optim -v
```

## Training-loop order

A typical step is:

1. build predictions and loss;
2. run `loss.backward()`;
3. optionally inspect/clip gradients;
4. call `optimizer.step()`;
5. clear gradients or rely on a fresh scalar graph that explicitly clears them.

## Important distinction

An optimizer can improve how training navigates the loss surface, but it does not fix incorrect data, leakage, a mismatched objective, or an unsuitable model architecture. Optimization quality and generalization quality are related but different concerns.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
