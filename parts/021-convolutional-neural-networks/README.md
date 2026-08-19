# Part 021 — Convolutional Neural Networks from First Principles

This companion part makes the core spatial operations behind convolutional neural networks inspectable before introducing a tensor framework. The implementation uses explicit Python loops on one-channel matrices so stride, padding, dilation, kernel traversal, and pooling remain visible.

## Covered concepts

- rectangular image/kernel validation;
- convolution output-size equations;
- symmetric zero padding;
- stride;
- dilation;
- deep-learning-style 2D cross-correlation;
- mathematical convolution with a flipped kernel;
- max pooling;
- average pooling;
- deterministic hand-checkable examples.

Reusable implementation: `src/neuralforge/convolution.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/021-convolutional-neural-networks/convolution_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/021-convolutional-neural-networks/convolution_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_convolution -v
```

## Cross-correlation versus convolution

Most deep-learning libraries call their sliding weighted-sum operation “convolution” even though the learned kernel is normally **not spatially flipped**. Mathematically that operation is cross-correlation. NeuralForge exposes both names so the distinction is explicit while preserving the convention learners will encounter in frameworks.

## Output shape

For one spatial dimension, NeuralForge uses:

`floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1`

The helper rejects an effective kernel that is larger than the padded input.

## Educational scope

This part deliberately stays single-channel and CPU-loop based. Multi-channel feature maps, trainable convolution filters, architecture patterns, residual connections, efficient mobile blocks, detection, and segmentation are introduced progressively in Parts 022–025.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
