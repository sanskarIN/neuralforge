# Part 022 — CNN Architecture Design: From LeNet to ResNet

Part 022 moves from individual convolution/pooling operations to complete CNN architecture reasoning. The companion code models layers as shape/parameter transformations, so learners can inspect why architectures shrink spatial resolution, increase channels, introduce residual projections, or transition from feature maps to classifiers.

## Covered concepts

- `(channels, height, width)` feature-shape propagation;
- exact convolution parameter counting;
- grouped convolution parameter counting;
- max/average pooling shape transitions;
- flatten and dense transitions;
- global average pooling;
- ResNet-style two-convolution basic blocks;
- automatic 1x1 projection shortcuts when residual shapes differ;
- trainable batch-normalization affine parameters in residual-block counts;
- architecture summaries for LeNet-5, AlexNet, VGG-11, and ResNet-18-style networks.

Reusable implementation: `src/neuralforge/cnn_architecture.py`.

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

## What the model counts

The generic convolution helper counts kernel weights plus optional biases. The ResNet basic-block helper models bias-free convolutions and includes trainable batch-normalization `gamma`/`beta` terms. Non-trainable running statistics are not counted as parameters.

The purpose is architectural reasoning rather than reproducing every framework implementation detail or historical preprocessing choice.

## Residual compatibility

A residual addition requires the main path and shortcut to produce the same shape. The companion model automatically counts a 1x1 projection shortcut when channel count changes or a block downsamples with stride greater than one.

## Design progression

- **LeNet** demonstrates early convolution/pooling/classifier composition.
- **AlexNet** demonstrates much larger channel capacity and a deep dense classifier.
- **VGG** demonstrates repeated small 3x3 convolutions in regular stages.
- **ResNet** demonstrates identity shortcuts and projection shortcuts that make much deeper stacks trainable.

Part 023 focuses on reducing compute and parameter cost for mobile/edge deployment.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
