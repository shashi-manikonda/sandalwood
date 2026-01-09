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
3.  **High-Level Wrapper (`CosyDA`)**: A Python class that manages the lifecycle of COSY variables (Indices).

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

Standard DA libraries often serialize coefficients into complex objects. Sandalwood uses specialized subroutines (`get_all_coeffs_flat`) that copy the internal COSY memory block directly into a pre-allocated NumPy array buffer.

* **Benefit:** Zero-copy overhead for large coefficient transfers.
* **Format:** The wrapper flattens the multi-dimensional exponent array into a 1D C-integer array, minimizing marshalling cost.

3. Static "Scratchpad" Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid the overhead of finding free memory slots for every intermediate calculation (e.g., `temp = a * b`), the `wrapper.f` module pre-allocates a persistent **Scratchpad** (Common Block `/DASCRATCH/`) of 20 variables.

Intermediate operations reuse these slots, eliminating the need to modify the global memory stack pointer (`IVAR`) for temporary arithmetic results.

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
