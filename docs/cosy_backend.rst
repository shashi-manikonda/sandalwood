.. _cosy_backend:

COSY Infinity Backend Architecture
==================================

The **COSY Backend** allows `sandalwood` to harness the raw speed and high-order capabilities of the **COSY Infinity** differential algebra library (written in Fortran 77). 

Unlike standard Python bindings (which often copy data excessively), this backend uses a custom **Direct Memory Bridge** designed for High-Performance Computing (HPC).

Architecture Overview
---------------------

The backend consists of three layers:

1.  **Fortran Core (`libcosy.so`)**: The compiled COSY Infinity library, patched with `wrapper.f` to expose a C-compatible ABI.
2.  **C-Types Bridge (`cosy_backend.py`)**: A thin Python layer that marshals pointers and integers directly to the shared library.
3.  **High-Level Wrapper (`CosyDA`)**: A Python class that manages the lifecycle of COSY variables using integer **Indices**, shielding the user from manual memory management. Ownership of an index can be safely transferred via :meth:`CosyDA.transfer_ownership`.

Memory Management Architecture
------------------------------

.. warning::
   **The COSY Fortran core is not thread-safe.**
   COSY Infinity uses global STATIC memory (Fortran COMMON blocks) for all calculations. The underlying Fortran routines are strictly single-threaded.

   **Sandalwood's Python layer** has been hardened for concurrent access:

   * ``CosyIndexPool.acquire`` and ``release`` are protected by a ``threading.RLock``, preventing duplicate-index allocation races when multiple threads hit the pool concurrently.
   * ``MultivariateTaylorFunction.initialize_mtf`` is protected by a module-level ``threading.RLock`` (``_INIT_LOCK``), ensuring class-state mutations are atomic.

   However, **concurrent calls into the Fortran library itself** (e.g., two threads each computing ``da * da`` through COSY at the same time) will still cause memory corruption. Only one thread should invoke COSY arithmetic at any given moment. Use a process-level lock or ``multiprocessing`` worker isolation if you need parallelism over COSY computations.

Effective memory management is critical when bridging Python's dynamic environment with COSY's static Fortran roots.

**CosyIndexPool: Object Recycling**
To prevent the overhead of frequent ``malloc/free`` cycles in the underlying stack, Sandalwood implements a **Robust Memory Pooling** (Object Pool Pattern).
The `CosyIndexPool` maintains a list of available COSY variable indices. When a calculation needs a temporary variable, it "acquires" an index from the pool in **O(1)** time. Once the calculation is complete, the index is "released" back to the pool for future reuse. This prevents "stack thrashing" and significantly improves performance in iterative batch operations.

Since v0.1.3 (``feat/backend-hardening``), both ``acquire`` and ``release`` are protected by an internal ``threading.RLock`` so that the pool itself is safe to access from multiple Python threads simultaneously. The Fortran core remains single-threaded (see warning above).

**CosyDA Ownership Transfer**
When a COSY operation produces a new ``CosyDA`` object, ownership of the underlying Fortran index must be transferred to the result wrapper exactly once — transferring twice would cause a double-free; failing to transfer would leak the index.

The ``transfer_ownership()`` method on ``CosyDA`` provides a single, explicit handoff:

.. code-block:: python

    res_da = some_operation()          # owned=True (will free on __del__)
    idx = res_da.transfer_ownership() # res_da.owned now False — will NOT free
    result = CosyMtfData(dimension, idx=idx, owned=True)  # new owner

This replaces the previous pattern of manually setting ``res_da.owned = False``, which was easy to forget or apply twice.

**Numerical Hygiene: DA_RESET**
Reusing memory indices requires strict hygiene. Before an index is acquired from the pool, it is subjected to a **Hard Reset** via the `DA_RESET` mechanism.
This Fortran-level routine explicitly:

* Sets the variable to a constant 0.0 using `DACON`.
* Re-initializes the internal type metadata (`NTYP`) to the base Differential Algebra type (`NDA`).
* Clears any stale high-order coefficients.

This ensures that every "new" variable is numerically clean, preventing intermediate results from leaking into subsequent calculations.

**Configuration: Pool Sizing**
The memory pool size can be tuned via the environment variable:

.. code-block:: bash

   export SANDALWOOD_COSY_POOL_SIZE=1024

The default value is **1024**. For massive batch operations or high-order compositions involving millions of intermediate terms, increasing this value (e.g., to 4096) can reduce acquisition latency, at the cost of higher static memory usage.

Specialized Physics Kernels
---------------------------

The COSY backend includes specialized kernels for high-performance physics simulations, accessible via **Hybrid Dispatch**.

**Biot-Savart Solver**
The `biot_savart_batch` function automatically dispatches to the most efficient kernel based on input types:

*   **Fast Path (Discrete Mode):** If inputs are `float64` arrays, the backend dispatches to a raw Fortran kernel (`COMPUTE_BIOT_SAVART_BATCH_FAST`). This bypasses the DA layer entirely, offering speeds comparable to compiled C/C++ code (~100x faster than Python loops).
*   **Parametric Path (MTF Mode):** If inputs are DA objects, the backend uses the high-order DA kernel. This allows for computing derivatives and Taylor    parametric equations and high-order derivatives.

.. code-block:: text

    [ Python User ]  <--  sandalwood.mtf
           |
    [ Python Backend ]  (cosy_backend.py)
           |  (ctypes / pointers)
           v
    [ Fortran Wrapper ] (wrapper.f / ISO_C_BINDING)
           |  (Common Blocks / Arrays)
           v
    [ COSY Core ]       (Static Memory Pool)

Memory Management: The "CosyScope"
----------------------------------

COSY Infinity uses a **Global Static Memory Pool** (Common Blocks `/DACOM/`). It allocates variables sequentially on a stack. This poses a challenge for Python, which uses a dynamic Garbage Collector.

If Python objects are deleted out of order, they would leave "holes" in the COSY stack, leading to memory fragmentation or corruption.

**The Solution: CosyScope**

We implemented a custom context manager, :class:`CosyScope`, which treats memory allocation like a stack frame in a compiler.

.. code-block:: python

    from sandalwood.backends.cosy.cosy_backend import CosyScope

    # State: IVAR=10, IMEM=500
    with CosyScope():
        # Allocation happens here
        x = CosyBackend.var(0) 
        y = x * x + 2
        # State: IVAR=12, IMEM=550
        
    # Exit Scope:
    # The backend strictly "rewinds" the internal pointers.
    # State: IVAR=10, IMEM=500
    # All temporary variables (x, y) are effectively freed instantly.

This approach is **O(1)** (constant time) for freeing memory, regardless of how many millions of temporary objects were created inside the loop.

Performance Optimizations
-------------------------

1. Batch Evaluation with OpenMP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluating a high-order polynomial at millions of points is the most expensive operation in tracking simulations. We offload this entirely to Fortran.

* **Function:** `eval_da_batch` (in `wrapper.f`)
* **Technique:** It flattens the pointer structure and uses **OpenMP** directives (`!$OMP PARALLEL DO`) to distribute the workload across all CPU cores.
* **Speedup:** ~50-100x compared to NumPy broadcasting for high orders.

2. Direct "Flat" Data Transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard DA libraries often serialize coefficients into complex objects. Sandalwood uses specialized subroutines (`get_all_coeffs_flat` and `cosy_set_coeffs`) that copy internal COSY memory blocks directly to/from pre-allocated NumPy array buffers.

* **Benefit:** Zero-copy overhead for large coefficient transfers (bidirectional).
* **Format:** The wrapper flattens the multi-dimensional exponent array into a 1D C-integer array, minimizing marshalling and transition costs.

Memory Configuration
--------------------

For advanced users, the COSY backend's internal memory limits (e.g., the size of the storage stack or the maximum number of variables) can be customized without modifying the core Fortran source files.

A configuration file is provided at ``src/sandalwood/backends/cosy/cosy_config.env``. You can modify the following parameters:

* **COSY_LMEM**: Length of the main storage stack (Default: 140,000,000).
* **COSY_LVAR**: Maximum number of variables (Default: 10,000,000).
* **COSY_LEA**: Maximum number of monomials. Increase this for extremely high-order calculations.
* **COSY_LNO**: Maximum supported Taylor order (Default: 99).
* **COSY_LNV**: Maximum number of variables (Default: 40).

### Compiler Support (Linux)

Sandalwood supports two main Fortran compilers on Linux:

*   **Intel Fortran (ifx)**: **The recommended default.** Recommended for maximum performance on Intel hardware. It provides superior auto-vectorization and highly optimized OpenMP performance.
*   **GNU Fortran (gfortran)**: The standard open-source fallback. Robust and widely available.

### Intel oneAPI Integration

If using `ifx`, you must ensure the Intel environment variables are loaded in your current shell before building. This is typically done by sourcing the ``setvars.sh`` script (Linux) or using the "Intel oneAPI Command Prompt" (Windows).

.. code-block:: bash

   # Linux example (run in shell or add to .bashrc)
   source /opt/intel/oneapi/setvars.sh

### Building the Backend

There are three ways to build the COSY backend components:

1. **Standard Python Install**
   When you run ``pip install -e .``, the ``setup.py`` script automatically detects your compiler, patches the memory limits, and builds the shared library.

2. **Standalone Shared Library Build**
   Use this for development or after changing memory parameters:

   .. code-block:: bash

      bash src/sandalwood/backends/cosy/compile_cosy.sh

3. **Raw COSY Binary Build ("cosy-raw")**
   For benchmarking against raw COSY Infinity or running legacy .FOX scripts, you can compile a standalone COSY executable:

   .. code-block:: bash

      bash scripts/benchmarks/benchmark.sh help

   The benchmark script also respects the ``COSY_COMPILER`` setting from the config file.

The build script automatically creates a temporary build directory, patches the Fortran sources with your new limits, and links the updated ``libcosy.so``. This ensures the original COSY source files in the repository remain un-modified.

3. Static "Scratchpad" Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid the overhead of finding free memory slots for every intermediate calculation (e.g., `temp = a * b`), the `wrapper.f` module pre-allocates a persistent **Scratchpad** (Common Block `/DASCRATCH/`) of 20 variables.

Intermediate operation wrappers (like `compute_da_add_const`) reuse these slots, eliminating the need to modify the global memory stack pointer (`IVAR`) for temporary arithmetic results. This reduces "stack churn" and improves cache locality.

4. Taylor Map Composition (`POLVAL`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Composing two high-order maps is mathematically equivalent to substituting one polynomial into another. Sandalwood uses a specialized wrapper for the COSY `POLVAL` routine.

* **Technique:** Instead of manual substitution in Python loops, we marshal the entire map to Fortran and use COSY's highly optimized, recursive substitution algorithm.
* **Benefit:** This is the most efficient way to perform symplectic tracking or map concatenation.

5. Vectorized Batch Arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For operations on large arrays of DA objects (e.g., adding two vectors of 10,000 Taylor series), we provide "Batch" variants.

* **Function:** `compute_da_add_batch`, `compute_da_mul_batch`, etc.
* **Technique:** These routines perform the loop over the array entirely within Fortran, avoiding 10,000 costly transitions between the Python interpreter and the shared library.

6. Novel: Stable Complex Arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

COSY's internal complex division and inversion can occasionally encounter stability issues with extremely small coefficients. Sandalwood includes **Patched Implementation** (``SANDALWOOD_CDMUI`` and ``SANDALWOOD_CDDCD``) that use a more robust normalization strategy.

* **Improvement:** Ensures high-precision results even in numerically sensitive regions of the complex plane.

Since v0.1.3, the Python-level ``CosyMtfData.inverse()`` and ``divide()`` guards have been tightened from exact ``== 0`` comparisons to tolerant ``abs(c0) < 1e-14`` checks, preventing false positives from floating-point rounding near zero.

Extending the Backend
---------------------

To add new COSY functions:

1.  **Modify `wrapper.f`**: Add a subroutine using `ISO_C_BINDING`.
    
    .. code-block:: fortran

       SUBROUTINE MY_FUNC(IDX_IN, IDX_OUT) BIND(C, NAME='my_func')
           USE ISO_C_BINDING
           INTEGER(C_INT) IDX_IN, IDX_OUT
           ! ... Call COSY internal ...
       END SUBROUTINE

2.  **Update `cosy_backend.py`**: Bind the function signature.

    .. code-block:: python

        bind_cosy_func("my_func", [POINTER(c_int), POINTER(c_int)])

3.  **Recompile**: Run `compile_cosy.sh`.
