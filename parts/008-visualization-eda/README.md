# Part 008 — Visualization and Exploratory Data Analysis

This companion module builds exploratory-analysis primitives without requiring a plotting framework, so learners can inspect the statistics and geometry behind simple visualizations before adopting larger libraries.

## Covered concepts

- linearly interpolated quantiles;
- numeric descriptive summaries;
- interquartile range (IQR);
- IQR-based outlier flags;
- fixed-bin histograms;
- robust handling of constant data;
- self-contained SVG scatter plots;
- safe escaping of plot titles.

The reusable implementation lives in `src/neuralforge/eda.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/008-visualization-eda/eda_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/008-visualization-eda/eda_demo.py
```

The demo writes `artifacts/part-008-training-curve.svg`. Generated `artifacts/` content should generally remain untracked unless it is intentionally needed as a small reference asset.

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_eda -v
```

## Interpretation rule

EDA can reveal patterns worth investigating, but a plot or correlation does not prove causation. Keep exploratory decisions separate from final test-set evaluation to avoid turning the test set into another tuning signal.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
