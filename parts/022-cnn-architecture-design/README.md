# Part 022 — CNN Architecture Design: From LeNet to ResNet

Part 022 moves from individual convolution operations to complete CNN feature-extractor design. The companion code analyzes architecture geometry and cost without requiring a deep-learning framework.

## Covered concepts

- convolution input/output shapes;
- trainable parameter counts;
- multiply-accumulate (MAC) estimates;
- grouped convolution geometry;
- pooling stages;
- receptive-field growth;
- effective output stride/jump;
- residual/skip shape compatibility;
- a compact LeNet-style feature extractor;
- architecture reasoning that prepares for residual networks.

Implementation: `src/neuralforge/cnn_architecture.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/022-cnn-architecture-design/architecture_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/022-cnn-architecture-design/architecture_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_cnn_architecture -v
```

## Why receptive field matters

Every convolution or pooling stage expands the region of the original input that can influence a later activation. The companion report tracks this growth together with the output jump so learners can reason about local detail versus broader context.

## Residual connections

An identity residual addition requires matching spatial dimensions and channels. If a main path changes resolution or channel count, a projection/strided skip path is usually needed before element-wise addition. The `residual_compatible` helper makes that shape constraint explicit.

## Cost estimates

MAC counts here describe convolution arithmetic only. Real latency also depends on memory access, kernel implementation, hardware, precision, batching, compiler/runtime behavior, and operator fusion. Part 023 builds on these distinctions for mobile/efficient vision.

---

**Official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
