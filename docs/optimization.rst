.. _optimization:

Performance Optimization in Sandalwood
======================================

One of the central challenges in Differential Algebra (DA) is the **Combinatorial Explosion**. As you increase the order of a Taylor series, the number of terms grows exponentially. Calculating `(x + y)^10` involves manipulating hundreds of coefficients; calculating `(x + y + z)^20` involves thousands.

Historically, this forced scientists to use rigid, compiled languages like Fortran (e.g., COSY Infinity) because Python's overhead was too high. ``sandalwood`` bridges this gap by implementing advanced high-performance computing (HPC) techniques directly in the Python backend.

This guide details the optimizations that allow ``sandalwood`` to perform within a factor of 5-10x of optimized C++ code, while retaining Python's flexibility.

The "Two-Language" Problem
--------------------------

In standard Python, every number is an "Object" (a box containing data + type info). Adding two numbers involves unboxing them, checking types, adding, and re-boxing the result.

* **The Bottleneck:** In a Taylor series multiplication, we perform millions of tiny additions. Doing this loop in standard Python would be 1000x slower than C++.
* **The Solution:** We must move the "Hot Loops" (the parts of code running millions of times) out of Python and into machine code, while keeping the high-level logic in Python.

1. "Dense Mode" Architecture
----------------------------

Standard symbolic libraries use a **Sparse** representation, storing terms as a list of `(exponent, coefficient)` pairs. To multiply two functions, they must generate every combination of exponents and then sort/merge them to combine like terms.

* **Why it's slow:** Sorting is an $O(N \log N)$ operation. As the polynomial grows, the time spent sorting exponents dominates the calculation.
* **Our Optimization:** When `initialize_mtf` is called, ``sandalwood`` pre-computes a **Multiplication Table**.

We map every possible exponent tuple (e.g., `x^2 y^1`) to a unique integer index. The multiplication logic becomes a simple "Lookup":

.. code-block:: python

    # Naive Sparse Approach (Slow)
    result = []
    for term_a in A:
        for term_b in B:
             new_exp = term_a.exp + term_b.exp  # Tuple addition
             new_coeff = term_a.coeff * term_b.coeff
             result.append((new_exp, new_coeff))
    result = sort_and_merge(result)  # Heavy sorting!

    # Sandalwood Dense Approach (Fast)
    # The 'Table' tells us exactly where the result goes. No sorting needed.
    target_index = Table[index_a, index_b]
    ResultArray[target_index] += CoeffA * CoeffB

This reduces the complexity from $O(N \log N)$ to **$O(1)$** per operation.

2. Numba JIT Compilation
------------------------

Even with the Dense Table, iterating through arrays in Python is slower than C. To fix this, ``sandalwood`` uses **Numba**, a Just-In-Time (JIT) compiler.

* **What it does:** At runtime, Numba translates our multiplication functions into **Optimized Machine Code** (LLVM IR), similar to what a C++ compiler produces.
* **Parallelization Strategy:** We implementing a **Map-Reduce** pattern with thread-local buffers to solve the race condition problem:

    1. **Allocation:** Use `get_num_threads()` to allocate a `(num_threads, output_size)` buffer.
    2. **Manual Chunking:** We explicitly partition the outer loop index `idx_a` into chunks `[start, end)` for each thread using `prange`.
    3. **Thread Isolation:** Thread `t` writes *only* to `buffer[t, :]`. This eliminates the need for atomic locks.
    4. **Reduction:** A final `np.sum(buffer, axis=0)` merges the partial results.

**Code Insight:** Use of `fastmath=True` allows the compiler to re-associate floating point operations for SIMD vectorization.

3. Lazy Materialization
-----------------------

A major hidden cost in Python libraries is **Object Creation**. Creating a new Python object for every intermediate step in a calculation like `x = (a + b) * (c + d)` is expensive.

* **The Optimization:** ``sandalwood`` uses **Lazy Evaluation**.
* **How it works:** When you multiply two MTFs, we compute the result as a raw array of numbers (Dense format) but **do not** convert it back to the user-friendly format (Exponents/Coefficients) immediately.
* **Benefit:** If you perform a chain of 100 operations, we skip the expensive conversion 99 times. We only "materialize" the full object when you actually ask to see the coefficients (e.g., printing the function).

4. Fast Evaluation with Power Caching
-------------------------------------

Evaluating a high-order polynomial at millions of points (e.g., particle tracking) implies computing `x^p` millions of times. `pow(x, p)` is slow.

* **Algorithm:** We use a **2D Thread-Local Power Cache**.
    1. For each point (in parallel), we pre-compute powers `[x^0, x^1, ... x^N]` for each variable dimension.
    2. These are stored in a simple array `cached_powers[dim][order]`.
    3. The evaluation loop simply looks up `cached_powers[d][idx]` instead of calling `pow`.
* **Memory Safety:** The `neval` method automatically detects large datasets and processes them in **Chunks** (default: 10,000 points) to keep the working set within CPU L1/L2 cache.


5. Vectorized Object Instantiation
----------------------------------

When working with large arrays of COSY results, standard Python loops for wrapping C pointers into Python objects can be a significant bottleneck.

**Anti-Pattern (Slow):**

.. code-block:: python

    # Slow loop creating objects one by one
    objects = []
    for idx in cosy_indices:
        objects.append(MultivariateTaylorFunction(idx))

**Best Practice (Fast):**

Use `from_cosy_indices` to perform bulk instantiation. This moves the loop to optimized C/Fortran or vectorized Python logic, reducing overhead.

.. code-block:: python

    # Fast bulk creation
    objects = MultivariateTaylorFunction.from_cosy_indices(cosy_indices)


Summary of Speedups
-------------------

Internal benchmarks show the impact of these optimizations:

+-------------------------+-----------------------+-----------------------+
| Operation               | Naive Python          | Optimized Sandalwood  |
+=========================+=======================+=======================+
| Multiplication (Ord 12) | ~500 ms               | **~45 ms** |
+-------------------------+-----------------------+-----------------------+
| Evaluation (1M pts)     | Crash (Out of Memory) | **Stable & Fast** |
+-------------------------+-----------------------+-----------------------+

6. Hybrid Dispatch & Fortran Fast Path
--------------------------------------

While COSY Infinity provides exact high-order derivatives, the overhead of Differential Algebra (DA) tracking can be excessive for simple scalar evaluations (e.g., calculating the magnetic field at millions of points for visualization).

*   **The Problem**: Using generic DA arithmetic for floating-point numbers incurs overhead from object wrapping, memory allocation, and sparse/dense array management.
*   **The Solution**: ``sandalwood`` implements a **Hybrid Dispatch** mechanism in ``CosyBackend``.

    1.  **Inspection**: The backend checks if the input coordinates are pure Python ``float/int`` or ``numpy.ndarray`` (scalars) versus ``MultivariateTaylorFunction`` objects.
    2.  **Fast Path (Scalars)**: If inputs are scalars, execution is routed to the new ``compute_biot_savart_batch_fast`` Fortran routine.
        *   **Implementation**: A pure ``DOUBLE PRECISION`` implementation of the Biot-Savart law.
        *   **Parallelism**: Uses **OpenMP** to parallelize the loop over target points.
        *   **Performance**: Avoids all DA overhead, achieving near-native Fortran speed (orders of magnitude faster than DA evaluation for scalars).
    3.  **General Path (DA)**: If inputs are MTFs, execution routes to the standard DA-enabled Fortran routines (`compute_biot_savart_batch`) to preserve derivative information.

This ensures the best of both worlds: high-performance visualization (using Fast Path) and high-precision optimization (using DA Path).

7. Phase 1 Optimizations (Vectorization & Batching)
---------------------------------------------------

Recent updates (Phase 1) focused on core arithmetic and composition bottlenecks.

Vectorized Horner's Method
~~~~~~~~~~~~~~~~~~~~~~~~~~
The `neval` kernel (`sandalwood.numba_kernels.evaluate_dense_kernel`) was refactored to use a **Block-Based Reduction** strategy.
*   **Block Processing**: Points are processed in blocks of 256 to maximize cache locality.
*   **Transposed Layout**: Powers are pre-computed in a `(dim, max_order, block_size)` layout, ensuring that the inner loop over `block_size` accesses contiguous memory, enabling SIMD vectorization.
*   **Result**: Significant speedup in large-scale evaluations.

Batch Composition
~~~~~~~~~~~~~~~~~
`TaylorMap.compose` previously performed term-by-term addition, incurring high overhead from repeated object creation and merging.
*   **Optimization**: Terms are now accumulated in a list `terms_to_sum`.
*   **Batch Add**: A new `_batch_add` method (in `taylor_function.py`) efficiently sums these terms in one pass (or using optimized reductions), minimizing intermediate object churn.

Fast-Path Dispatch
~~~~~~~~~~~~~~~~~~
Arithmetic operator dispatch (``__add__``, ``__mul__``, etc.) uses a simple
``if self._IMPLEMENTATION == "cosy"`` branch inside each dunder method.
This avoids any global class mutation at initialization time and is safe
from concurrent threads. The branch is predictable (same condition throughout
the lifetime of a session), so modern CPUs can speculate it with near-zero
overhead.

COSY Scalar Operators
~~~~~~~~~~~~~~~~~~~~~
`CosyMtfData` (the data holder for COSY backend) now supports direct operator overloading (e.g., `da + 1.0`), removing the need for explicit constant wrapping in standard Python arithmetic.

8. Phase 2 Optimizations (Solver & Batching)
--------------------------------------------

The second phase targeted the integration layer between the high-level solvers and the backend kernels.

Structure of Arrays (SoA)
~~~~~~~~~~~~~~~~~~~~~~~~~
In `em-simulation-platform`, calculations were refactored to strictly enforce **Structure of Arrays (SoA)** memory layout. 
Instead of processing lists of `FieldVector` objects (Array of Structures), the solvers now manipulate contiguous arrays for `Bx`, `By`, and `Bz` components independently.

*   **Benefit**: Improved cache locality and SIMD vectorization potential during Biot-Savart integration.
*   **Result**: The `calculate_b_field` function now returns a SoA-optimized `VectorField` object directly.

Batch COSY Operations
~~~~~~~~~~~~~~~~~~~~~
To reduce the overhead of switching between Python and C for every single arithmetic operation, we implemented **Batch Dispatch** in `CosyBackend`.

*   **`linear_combination`**: Computes $\sum c_i \cdot x_i$ entirely within the C kernel. This is used for `TaylorMap.compose` and aggregating contributions from multiple source segments.
*   **`batch_arithmetic`**: Performs element-wise operations (Add, Sub, Mul, Div) on lists of DA objects in a single call.
*   **Optimization Strategy**: Due to stability issues with the native `da_lin_comb` COSY function, we implemented a robust **Iterative Fallback** at the C-interface level. This loops over inputs using direct low-level C functions (`compute_da_add`, `compute_da_mul_const`) without returning control to Python, preserving performance while ensuring correctness.

9. Final Benchmark Results
--------------------------

We benchmarked the `RingCoil` B-field calculation (Biot-Savart Law) to measure the cumulative impact of these optimizations. The comparison is between the optimized Python backend and the new SoA-optimized COSY backend.

**Test Case**: 1,000 source segments, Order 1 calculation.

+----------------+--------------+--------------+-------------+
| Field Points   | Python (s)   | COSY (s)     | Speedup     |
+================+==============+==============+=============+
| 100            | 0.0157       | 0.0112       | **1.41x**   |
+----------------+--------------+--------------+-------------+
| 10,000         | 1.1031       | 0.0074       | **148.68x** |
+----------------+--------------+--------------+-------------+
| 100,000        | 11.1953      | 0.0243       | **461.23x** |
+----------------+--------------+--------------+-------------+


**Conclusion**: The combination of SoA layout, Fast-Path dispatch, and Batch processing has yielded a **~460x speedup** for large-scale simulations.

10. Map Composition Optimization (Phase 4)
------------------------------------------

Map composition (substituting one map into another: $F(G(x))$) is computationally expensive because it requires expanding high-order polynomials raised to high powers.

*   **The Bottleneck**: In sparse mode, computing $(a + b + \dots)^n$ generates an explosion of intermediate terms before simplification.
*   **The Solution**: We implemented a **Hybrid Strategy** that combines high-level topological logic with **Just-In-Time (JIT) compiled kernels** using Numba.

**Algorithm Design:**

1.  **Pre-computation**: Before the main loop, we compute all necessary powers of the inner map components $(G_1(x)^p, G_2(x)^p, \dots)$ and store them in a contiguous 3D dense array `(dim, max_order+1, n_terms)`.
2.  **Parallel Accumulation**: We iterate over the terms of the outer map $F$ in parallel.
    *   For each term $c \cdot x_1^{p_1} x_2^{p_2} \dots$, we fetch the pre-computed dense vectors.
    *   We multiply them using the global **Multiplication Table** kernel (`dense_mul`).
    *   The result is accumulated into **Thread-Local Buffers** to avoid race conditions.
3.  **Result**: The final dense array is converted back to a sparse MTF only once at the end.

**Performance Impact**:
Benchmarks show a **3x - 5x speedup** compared to the pure Python implementation for typical 2D and 3D maps at orders 4-8. This optimization is automatically engaged when the multiplication table is available (dense mode).
