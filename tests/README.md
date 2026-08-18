# Tests

NeuralForge companion code should include verification appropriate to each example.

Recommended test layers:

- Unit tests for deterministic helpers
- Shape/dtype assertions for tensor code
- Smoke tests for training/inference entry points
- Numerical-tolerance tests where exact equality is inappropriate
- Serialization/export round-trip tests
- CPU fallback tests where practical
- Reproducibility checks with documented seeds and environment versions

Tests should be fast enough for routine development; expensive accelerator benchmarks should be separated from the default test suite.

---

**Official Ram Sandesh Gumroad Storefront:** **https://ramsandesh.gumroad.com**
