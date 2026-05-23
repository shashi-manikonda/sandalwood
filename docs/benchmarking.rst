.. _benchmarking:

Benchmarking and Performance Analysis
=====================================

To maintain its performance edge, ``sandalwood`` includes a comprehensive benchmarking suite. This toolkit allows developers to measure the overhead of the Python-Fortran bridge, compare backend implementations, and analyze scaling behavior across high-order Taylor series.

Quick Start
-----------

The easiest way to run benchmarks is using the included shell wrapper:

.. code-block:: bash

   # Run a standard comparison of individual operations (Order 8, 4 Variables)
   ./scripts/benchmarks/benchmark.sh ops

   # Run a three-way comparison including "Raw COSY" (direct Fortran timing)
   ./scripts/benchmarks/benchmark.sh raw

   # Run a full parametric sweep with HTML report generation
   ./scripts/benchmarks/benchmark.sh full

Benchmarking Modes
------------------

The suite supports several specialized modes:

1. **Operations (`ops`)**:
   Compares the **Python (Numba)** backend against the **COSY (Fortran)** backend for basic arithmetic (``+``, ``-``, ``*``, ``/``) and elementary functions (``sin``, ``exp``, etc.).

2. **Three-Way Comparison (`raw`)**:
   Adds a third competitor: **Raw COSY**. This executes the logic inside a standalone Fortran binary, measuring the absolute minimum time required by the COSY core. This reveals the "Bridge Overhead" introduced by CTypes and Python marshalling.

   .. note::
      The ``raw`` and ``full`` benchmarking modes execute standalone COSY scripts and require the proprietary ``cosy.fox`` file (the main COSY macro package). Since this file is proprietary, it is not distributed with Sandalwood. You must place a copy of your licensed ``cosy.fox`` in ``scripts/benchmarks/COSY.fox`` or set the ``SANDALWOOD_COSY_SRC`` environment variable pointing to your COSY source directory before running these benchmarks.

3. **Batch Evaluation (`batch`)**:
   Tests the performance of the ``neval()`` method. This highlights the efficiency of the OpenMP-optimized batch evaluator in the COSY backend versus the vectorized NumPy/PyTorch implementations.

4. **Parametric Sweep (`full`)**:
   Automatically runs a battery of tests across varying orders (2 to 10) and dimensions (4 to 6). This is the primary tool for detecting performance regressions in high-order logic.

Methodology
-----------

Accurate benchmarking of Differential Algebra requires handling extremely small timescales and hardware jitter. ``sandalwood`` employs the following strategies:

Timing Strategy
~~~~~~~~~~~~~~~

* **CPU vs Wall Time:** We use ``time.process_time()`` to measure only the time spent by the CPU on the current process, ignoring system-level context shifts.
* **Warmup & Repeats:** Every benchmark includes a "Warmup" phase to trigger JIT compilation (Numba) or cache heating. Results are averaged over multiple ``repeats`` to filter out noise.
* **Internal Fortran Timing:** For "Raw COSY", we use the internal COSY ``CPUSEC`` function. This ensures we measure only the mathematical execution time, excluding binary startup and disk I/O.

Memory Profiling
~~~~~~~~~~~~~~~~

When the ``--memory`` flag is used, the suite utilizes Python's ``tracemalloc`` to capture **Peak Memory Usage**. This is critical for monitoring the "Combinatorial Explosion" of high-order Taylor series and ensuring that the Dense Mode tables don't exceed available RAM.

Bridge Overhead Mitigation
~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure fairness when comparing against Raw Fortran:
* Expressions are pre-compiled into Python Bytecode (Lambdas) before the timing loop starts to eliminate ``eval()`` overhead.
* Large arrays are pre-allocated and reused to avoid garbage collection interference.

Reporting Tools
---------------

The benchmarking suite generates multiple reports:

* **Markdown Tables:** Simplified tables printed to the console and saved in ``scripts/benchmarks/artifacts/``.
* **HTML Reports:** The ``full`` mode generates a rich dashboard including:
    - System specifications (CPU, RAM, OS).
    - Interactive, sortable performance tables.
    - Speedup ratios (e.g., "COSY is 12.5x faster than Python").
    - Embedded plots showing scaling behavior.

Extending Benchmarks
--------------------

To add a new operation to the benchmark suite, modify the ``generate_benchmark_cases`` function in ``scripts/benchmarks/run.py``. You must provide a Python expression and its equivalent COSY-compatible string.

.. code-block:: python

   cases.append(("NewOp", "mtf.special_op(x)", "SPECIAL(DA(1))"))
