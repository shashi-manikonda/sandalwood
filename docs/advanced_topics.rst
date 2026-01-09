.. _theory:

Advanced Topics
======================

Backends and Performance
------------------------

`sandalwood` is designed for high performance by abstracting key numerical
operations to a flexible backend system. This allows the library to leverage
optimized libraries like NumPy and PyTorch.

### The Backend System

The `backend.py` module defines a set of static methods for common
tensor operations (e.g., `power`, `prod`, `dot`). The `neval`
method in `MultivariateTaylorFunction` automatically selects the appropriate
backend based on the input type.

### NumPy Backend

This is the default backend and is used when the input to `neval` is a
NumPy array. All operations are performed using NumPy's highly optimized
C and Fortran libraries, providing excellent performance for CPU-based computation.

### PyTorch Backend

When a PyTorch `Tensor` is passed to `neval`, `sandalwood` automatically
switches to the PyTorch backend. This provides two key advantages:

* **GPU Acceleration:** PyTorch's native CUDA support allows operations to be
    executed on an NVIDIA GPU, dramatically speeding up computations on large
    batches of data.
* **Vector and Matrix Operations:** PyTorch's backend is specifically
    optimized for deep learning, making it exceptionally fast for the types of
    vectorized operations used in `sandalwood`.

Future Work
-----------

This section outlines potential future directions for `sandalwood`, ranging from
incremental improvements to more ambitious features.


### Expanded Functionality

* **Automatic Differentiation (AD):** Implement support for AD to allow for
  the computation of gradients, Jacobians, and Hessians of Taylor maps.
* **Advanced Elementary Functions:** Add support for more advanced
  elementary functions, such as Bessel functions, gamma functions, and error
  functions.
* **Non-linear Equation Solver:** Implement a solver for systems of
  non-linear equations using Taylor series methods (e.g., Newton's method
  with high-order corrections).

### Improved User Experience

* **Symbolic API:** Develop a more intuitive API for creating and
  manipulating Taylor maps, perhaps with a more "symbolic" feel.
* **Example Gallery:** Create a "gallery" of examples in the documentation
  showcasing real-world applications of `sandalwood` in physics, engineering,
  and other fields.
* **Improved Error Messages:** Add more detailed error messages and warnings
  to help users debug their code.
