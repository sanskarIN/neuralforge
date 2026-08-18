# Part 011 — Computational Graphs and Automatic Differentiation

This companion part implements a tiny scalar reverse-mode automatic-differentiation engine from scratch. The goal is to make computational graphs and the chain rule visible before using framework autograd systems.

## Covered concepts

- scalar computational-graph nodes;
- parent/operation tracking;
- topological ordering;
- local derivative rules;
- reverse-mode gradient propagation;
- gradient accumulation through shared subgraphs;
- addition, subtraction, multiplication, division, and powers;
- `exp`, `log`, `tanh`, ReLU, and sigmoid;
- explicit gradient clearing;
- simple graph node/edge/operation summaries.

The implementation lives in `src/neuralforge/autograd.py`.

## Run the demo

```bash
PYTHONPATH=src python parts/011-autograd/autograd_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/011-autograd/autograd_demo.py
```

## Run tests

```bash
PYTHONPATH=src python -m unittest tests.test_autograd -v
```

## Why reverse mode matters

A training loss is usually one scalar produced from many parameters. Reverse-mode autodiff efficiently propagates derivatives from that scalar back to all parameters, which is the pattern needed by neural-network training.

## Educational scope

This scalar engine is deliberately small. Production frameworks operate on tensors, optimize graph execution, support devices and mixed precision, and handle far more operations. The purpose here is to understand the mechanism that those systems automate.
