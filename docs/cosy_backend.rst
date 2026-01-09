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
3.  **High-Level Wrapper (`CosyDA`)**: A Python class that manages the lifecycle of COSY variables using integer **Indices**, shielding the user from manual memory management.

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

COSY's internal complex division and inversion can occasionally encounter stability issues with extremely small coefficients. Sandalwood includes **Patched Implementation** (`SANDALWOOD_CDMUI` and `SANDALWOOD_CDDCD`) that use a more robust normalization strategy.

* **Improvement:** Ensures high-precision results even in numerically sensitive regions of the complex plane.

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
