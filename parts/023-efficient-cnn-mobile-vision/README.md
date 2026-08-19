# Part 023 — Efficient CNNs & Mobile Vision

Part 023 turns “lightweight CNN” into measurable engineering trade-offs. The companion implementation estimates trainable convolution weights, multiply-accumulate operations (MACs), output activation size, width/resolution scaling, and MobileNetV2-style inverted residual blocks without requiring a deployment framework.

## Covered concepts

- standard convolution parameter/MAC cost;
- depthwise spatial convolution;
- 1x1 pointwise channel mixing;
- depthwise-separable convolution cost;
- FP32 parameter/output memory estimates;
- MobileNet-style channel divisibility;
- width multipliers;
- resolution multipliers;
- inverted residual expansion-depthwise-projection blocks;
- residual-connection eligibility;
- parameter and MAC reduction ratios.

Reusable implementation: `src/neuralforge/mobile_vision.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/023-efficient-cnn-mobile-vision/efficiency_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/023-efficient-cnn-mobile-vision/efficiency_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_mobile_vision -v
```

## Why MACs matter

Parameter count describes model storage but does not fully describe runtime cost. A convolution reuses each kernel weight at many spatial locations, so spatial resolution strongly affects compute. The helper reports MACs separately from parameters to make that distinction visible.

## Depthwise-separable intuition

A standard `K×K` convolution learns spatial and channel mixing together. A depthwise-separable block splits the work:

1. one `K×K` spatial filter per input channel;
2. one `1×1` pointwise convolution to mix channels.

For many channel configurations this dramatically reduces parameters and MACs while preserving a useful spatial/channel factorization.

## Inverted residuals

The MobileNetV2-style estimator models:

1. 1×1 expansion;
2. depthwise spatial convolution;
3. 1×1 linear projection;
4. a skip connection only when stride is 1 and input/output channels match.

The estimator intentionally counts convolution weights, not hardware-specific latency. Real latency also depends on memory access, kernel implementation, accelerator support, quantization, operator fusion, and device constraints.

Part 024 moves from image classification features to object-detection geometry and post-processing.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
