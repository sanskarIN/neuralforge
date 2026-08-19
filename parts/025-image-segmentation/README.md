# Part 025 — Image Segmentation: From Pixels to Panoptic Understanding

Part 025 introduces segmentation through explicit per-pixel evaluation rather than hiding metric behavior behind a framework. The companion implementation supports binary and multiclass label masks, ignore labels, probability thresholding, IoU/Dice metrics, and a small deterministic post-processing filter.

## Covered concepts

- rectangular non-negative integer label masks;
- boolean masks;
- binary true-positive/false-positive/false-negative/true-negative counts;
- binary precision, recall, accuracy, IoU/Jaccard, and Dice;
- empty-positive-class behavior;
- multiclass confusion matrices;
- per-class IoU, Dice, precision, and recall;
- mean IoU with absent-class control;
- ignore-label handling;
- probability-to-mask thresholding;
- local majority-filter mask smoothing.

Reusable implementation: `src/neuralforge/segmentation.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/025-image-segmentation/segmentation_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/025-image-segmentation/segmentation_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation -v
```

## Metric conventions

For a class with no predicted or target positive pixels, the binary IoU/Dice helpers return `1.0`: the two masks agree perfectly for that class. For multiclass mean IoU, absent classes are excluded by default so they do not inflate an image-level mean; `include_empty=True` can be selected explicitly.

## Ignore labels

Dataset annotations often contain pixels that should not contribute to evaluation. When `ignore_label` is supplied, those **target** pixels are excluded before confusion counts are accumulated.

## Majority filtering

The local majority filter is intentionally simple and dependency-free. At each pixel, it chooses the most common label in an odd-sized neighborhood clipped to image boundaries. If multiple labels tie and the center label is one of the winners, the center is preserved; otherwise the smallest winning label is chosen deterministically.

## Scope boundary

This part establishes segmentation metrics and mask mechanics. Production semantic, instance, and panoptic systems additionally require model architectures, dataset pipelines, augmentation, multi-scale inference, instance association, benchmark-specific void/crowd handling, and task-specific evaluation protocols.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
