# Part 025 — Image Segmentation: From Pixels to Panoptic Understanding

Part 025 finishes NeuralForge's first vision milestone by evaluating predictions at pixel level and connecting semantic segmentation metrics to panoptic-quality concepts.

## Covered concepts

- semantic label masks;
- confusion matrices with target rows / prediction columns;
- ignored labels;
- pixel accuracy;
- per-class true positives, false positives, and false negatives;
- intersection over union (IoU);
- Dice score;
- mean IoU and mean Dice across present classes;
- binary mask IoU/Dice;
- panoptic quality (PQ);
- segmentation quality (SQ);
- recognition quality (RQ).

Implementation: `src/neuralforge/segmentation.py`.

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

## Semantic segmentation

Semantic segmentation assigns a class label to each evaluated pixel. The confusion matrix then provides enough information to compute pixel accuracy, class IoU, Dice, and macro averages over classes that actually occur in the evaluated prediction/target pair.

## Ignore labels

Datasets often contain void/unlabeled pixels. If an `ignore_index` is configured, both the target and prediction at that pixel are excluded from class inference and metric accumulation. This avoids treating intentionally unevaluated regions as prediction errors.

## Panoptic quality

Panoptic quality combines two ideas:

- **SQ** — how well matched segments overlap;
- **RQ** — how reliably segments are recognized/matched.

`PQ = SQ × RQ` in the companion helper. A benchmark implementation must still follow the dataset's exact instance-matching and category conventions.

## Benchmark caution

Do not compare numbers across papers or datasets unless preprocessing, label mappings, ignored regions, resizing/cropping rules, class sets, averaging conventions, and official evaluation code are compatible.

---

**Official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
