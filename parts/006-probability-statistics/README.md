# Part 006 — Probability and Statistics for Learning Systems

This companion module implements statistical building blocks used when describing data, uncertainty, evaluation noise, and sampling variation in machine-learning systems.

## Covered concepts

- arithmetic mean;
- population and sample variance;
- standard deviation;
- covariance and Pearson correlation;
- normal probability density;
- Bernoulli log-likelihood;
- seeded percentile bootstrap intervals for the mean;
- input validation for constant, mismatched, empty, and non-finite data.

The implementation is dependency-free and lives in `src/neuralforge/statistics.py`.

## Run the demo

From the repository root:

```bash
PYTHONPATH=src python parts/006-probability-statistics/statistics_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/006-probability-statistics/statistics_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_statistics -v
```

## Reproducibility note

Bootstrap results use a local seeded random generator. Reusing the same data, seed, confidence level, and resample count produces the same interval for this implementation.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
