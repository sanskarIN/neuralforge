# Part 023 — Efficient CNNs & Mobile Vision

Part 023 turns CNN accuracy discussions into deployment-aware engineering by estimating the arithmetic and parameter cost of common mobile-vision building blocks.

## Covered concepts

- standard convolution parameter/MAC cost;
- depthwise convolution;
- pointwise `1x1` convolution;
- depthwise-separable convolution;
- inverted residual blocks;
- expansion ratios;
- stride-dependent output geometry;
- width multipliers and channel rounding;
- FP32/FP16/INT8-style model-size estimates;
- why parameter count, MACs, latency, memory, and energy are related but not identical.

Implementation: `src/neuralforge/mobile_vision.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/023-efficient-mobile-vision/mobile_cost_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/023-efficient-mobile-vision/mobile_cost_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_mobile_vision -v
```

## Depthwise-separable intuition

A standard `KxK` convolution mixes spatial and channel information in one operation. A depthwise-separable block splits that work into:

1. one spatial filter per input channel; then
2. a `1x1` pointwise convolution that mixes channels.

This can dramatically reduce parameters and MACs, especially when channel counts are large.

## Cost is not latency

A lower MAC count does not guarantee proportionally lower device latency. Runtime performance depends on memory traffic, tensor layouts, operator fusion, kernels, vectorization, accelerator support, precision, thermal limits, and framework/compiler overhead.

Use these helpers as architecture estimates, then measure real target-device latency and energy before making deployment claims.

---

**Official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
