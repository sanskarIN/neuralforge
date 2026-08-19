# Part 016 — Loss Functions and Output-Layer Design

Part 016 connects a learning task to the final model output and the objective used for training. The goal is to avoid common mismatches such as applying a probability activation twice or using a classification loss for an incompatible target format.

## Implemented concepts

- mean squared error;
- mean absolute error;
- Huber loss;
- numerically stable softplus;
- binary cross-entropy directly from logits;
- stable log-sum-exp;
- categorical cross-entropy directly from logits;
- mean multiclass cross-entropy over a batch;
- task/output/objective recommendations for regression, binary classification, and multiclass classification.

Reusable implementation: `src/neuralforge/losses.py`.

## Logits-first design

For binary and multiclass classification, the reusable objectives accept **raw logits**. That lets the loss combine the probability transformation and logarithm in a numerically safer form.

Typical pairings in this companion module are:

| Task | Final training output | Objective | Inference |
|---|---|---|---|
| Regression | linear scalar | MSE or Huber | use scalar directly |
| Binary classification | one logit | BCE with logits | sigmoid then threshold |
| Multiclass classification | one logit per class | categorical CE with logits | softmax / argmax |

## Run the demo

```bash
PYTHONPATH=src python parts/016-loss-functions-output-design/loss_design_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/016-loss-functions-output-design/loss_design_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_losses -v
```

## Stability rule

Avoid computing a probability, clipping it manually, and then taking a logarithm when a stable logits-based objective is available. Keeping the raw logits lets the objective use stable softplus/log-sum-exp identities while preserving correct gradients.

---

**NeuralForge / Ram Sandesh official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
