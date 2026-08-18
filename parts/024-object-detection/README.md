# Part 024 — Object Detection from First Principles to Real-Time Systems

Part 024 introduces the geometry and post-processing logic shared by many object detectors without tying the learner to one specific model family.

## Covered concepts

- `x1, y1, x2, y2` bounding boxes;
- center/width/height coordinate conversion;
- box width, height, and area;
- intersection over union (IoU);
- IoU matrices;
- score ordering;
- non-maximum suppression (NMS);
- class-aware NMS;
- one-to-one detection/ground-truth matching;
- IoU matching thresholds;
- true positives, false positives, false negatives;
- precision and recall.

Implementation: `src/neuralforge/detection.py`.

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

## Why NMS exists

Dense detectors can produce several high-scoring boxes around the same object. NMS keeps the strongest prediction and suppresses sufficiently overlapping lower-scoring boxes, normally within the same class.

## Evaluation boundary

The included evaluator intentionally teaches one IoU threshold and score-ordered one-to-one matching. Production benchmark metrics such as COCO-style AP average performance across confidence/IoU settings and add dataset-specific conventions. Those metrics should be implemented against the exact benchmark specification rather than approximated casually.

## Real-time deployment

A detector's end-to-end speed includes preprocessing, model execution, box decoding, NMS/post-processing, data transfer, rendering, and application overhead. Model FLOPs/MACs alone are not a complete real-time latency measurement.

---

**Official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
