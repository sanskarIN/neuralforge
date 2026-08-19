# Part 024 — Object Detection from First Principles to Real-Time Systems

Part 024 focuses on the geometry and post-processing that turn raw detector outputs into evaluated object predictions. The implementation is framework-independent so bounding-box conventions, overlap, suppression, and one-to-one matching remain explicit.

## Covered concepts

- axis-aligned `(x1, y1, x2, y2)` boxes;
- center/size `(cx, cy, width, height)` conversion;
- positive-area validation;
- clipping boxes to image bounds;
- intersection over union (IoU);
- scored/class-labeled detections;
- class-aware non-maximum suppression (NMS);
- class-agnostic NMS;
- maximum retained detection limits;
- greedy one-to-one prediction/ground-truth matching;
- true positives, false positives, false negatives;
- precision and recall.

Reusable implementation: `src/neuralforge/detection.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/024-object-detection/detection_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/024-object-detection/detection_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_detection -v
```

## Coordinate convention

A `BoundingBox(x1, y1, x2, y2)` uses continuous edge coordinates. Width is `x2 - x1` and height is `y2 - y1`; therefore both must be positive. This avoids the inclusive-pixel `+1` convention sometimes found in legacy implementations.

## NMS behavior

NMS sorts detections by score, keeps the highest-scoring candidate, then suppresses lower-scoring boxes whose IoU exceeds the configured threshold. By default suppression happens only within the same class. Class-agnostic NMS can be selected explicitly.

## Evaluation behavior

`match_detections` considers predictions in descending score order and lets each ground-truth object match at most once. An overlapping duplicate prediction therefore becomes a false positive after the corresponding object has already been matched.

This is a compact foundation rather than a complete benchmark implementation. Dataset-level average precision/mAP additionally requires score-threshold sweeps, class aggregation, benchmark-specific IoU thresholds, ignore regions, crowd rules, and other protocol details.

Part 025 shifts from bounding boxes to per-pixel image segmentation.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
