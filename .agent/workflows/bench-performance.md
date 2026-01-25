---
description: Performance and Regression Benchmarking
---

Verify the impact of core algebraic changes on throughput.

1. Run standard unit tests to ensure correctness:
   ```bash
   pytest
   ```
2. // turbo
   Run core multiplication benchmarks:
   ```bash
   python scripts/benchmarks/benchmark_multiplication.py
   ```
3. Run map composition benchmarks:
   ```bash
   python scripts/benchmarks/benchmark_composition.py
   ```
4. Compare results against the benchmark baseline in `docs/optimization.rst`.
