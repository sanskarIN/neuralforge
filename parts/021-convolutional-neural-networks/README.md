# Part 021 — Convolutional Neural Networks from First Principles

Part 021 begins NeuralForge's computer-vision block by making the geometry of a CNN explicit before introducing framework layers.

## Covered concepts

- 2D feature maps and kernels;
- deep-learning cross-correlation versus mathematical convolution;
- stride;
- explicit and SAME padding;
- dilation and effective kernel size;
- convolution output geometry;
- bias terms;
- max pooling;
- average pooling;
- rectangular/finite-value validation.

The reusable dependency-free implementation lives in `src/neuralforge/convolution.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/021-convolutional-neural-networks/cnn_primitives_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/021-convolutional-neural-networks/cnn_primitives_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_convolution -v
```

## Why the default is cross-correlation

Many deep-learning libraries call their operation "convolution" even though the kernel is not flipped. NeuralForge follows that convention by default because learned kernels do not require a hand-designed orientation. Set `flip_kernel=True` when you specifically want mathematical convolution.

## Educational scope

This implementation handles a single 2D image and a single 2D kernel so every index calculation remains inspectable. Production CNN layers add channels, batches, many filters, accelerators, optimized kernels, automatic differentiation, and mixed precision.

Those larger concerns are introduced incrementally in later vision parts rather than hidden inside Part 021.

---

**Official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
