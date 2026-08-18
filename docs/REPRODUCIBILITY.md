# Reproducibility Policy

NeuralForge examples and labs should be reproducible enough that readers can understand why two runs differ and how to recreate a reported result.

## Minimum experiment record

For experiments that report metrics, record when applicable:

- Python and framework versions;
- operating system and accelerator type;
- dataset name, version, split, and preprocessing;
- model/configuration identifier;
- random seed policy;
- optimizer, learning rate, batch size, and training duration;
- evaluation metric definition;
- checkpoint or artifact provenance;
- known nondeterministic operations.

## Randomness

Set explicit seeds where doing so is meaningful, but do not claim bit-for-bit determinism unless it has actually been verified for the target hardware/software stack.

## Data

Do not commit restricted or personally sensitive datasets. Prefer download/build instructions and checksums for redistributable data. Record licenses and source provenance.

## Models and large artifacts

Large weights should normally be referenced by a stable upstream source and checksum rather than committed directly. Any redistributed weight must have compatible rights and attribution.

## Environment files

Part-specific code may provide pinned or bounded dependency files. Avoid unnecessary global pinning when framework compatibility differs across parts.

## Result reporting

Separate measured results from illustrative/example values. When hardware materially affects latency, throughput, memory, or energy use, state the tested hardware.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
