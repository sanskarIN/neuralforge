# Part 016 — Loss Functions and Output-Layer Design

Part 016 turns loss selection into an explicit engineering decision instead of treating every task as "add an activation and compute a loss."

## Implemented concepts

- mean-squared error (MSE);
- mean-absolute error (MAE);
- Huber loss with a configurable transition point;
- numerically stable softplus;
- binary cross-entropy **from logits**;
- multiclass cross-entropy **from logits** with maximum-logit shifting;
- differentiable gradients through the scalar autodiff engine;
- recommended output/loss pairings for regression, binary classification, and multiclass classification.

Reusable implementation: `src/neuralforge/losses.py`.

## Recommended pairings

- **Regression:** linear output + MSE/MAE/Huber depending on the error behavior you want.
- **Binary classification:** one linear logit + binary cross-entropy with logits. Apply sigmoid when you need a probability, not before the training loss.
- **Multiclass classification:** one linear logit per class + multiclass cross-entropy from logits. A separate training-time softmax is unnecessary.

Computing classification losses directly from logits avoids fragile expressions such as `log(sigmoid(z))` for very large positive/negative values.

## Run the demo

```bash
PYTHONPATH=src python parts/016-loss-functions/loss_design_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/016-loss-functions/loss_design_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_losses -v
```

## Selection principle

The training objective should match the target representation and model output semantics. A numerically convenient loss is useful only when it also represents the task you intend to optimize.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
