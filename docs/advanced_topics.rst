.. _theory:

Advanced Topics
======================

Backends and Performance
------------------------

`sandalwood` is designed for high performance by abstracting key numerical
operations to a flexible backend system. This allows the library to leverage
optimized libraries like NumPy and PyTorch.

The Backend System
~~~~~~~~~~~~~~~~~~

The ``backend.py`` module defines a set of static methods for common
tensor operations (e.g., ``power``, ``prod``, ``dot``). The ``neval``
method in ``MultivariateTaylorFunction`` automatically selects the appropriate
backend class based on the input array type by calling ``get_backend(array)``.

``get_backend`` returns the backend **class** (not an instance). Because
all backend methods are ``@staticmethod``, the class itself is the callable,
so no object is constructed on each dispatch call.

NumPy Backend
~~~~~~~~~~~~~

This is the default backend and is used when the input to ``neval`` is a
NumPy array. All operations are performed using NumPy's highly optimized
C and Fortran libraries, providing excellent performance for CPU-based
computation.

Memory semantics
^^^^^^^^^^^^^^^^

``NumpyBackend.from_numpy(a, copy=True)`` (the default) always returns a
fresh copy. Pass ``copy=False`` only when you need a zero-cost view and are
certain no in-place mutation will occur.

PyTorch Backend
~~~~~~~~~~~~~~~

When a PyTorch ``Tensor`` is passed to ``neval``, `sandalwood` automatically
switches to the PyTorch backend. This provides two key advantages:

* **GPU Acceleration:** PyTorch's native CUDA support allows operations to be
  executed on an NVIDIA GPU, dramatically speeding up computations on large
  batches of data.
* **Vector and Matrix Operations:** PyTorch's backend is specifically
  optimized for deep learning, making it exceptionally fast for the types of
  vectorized operations used in `sandalwood`.

Device and gradient safety
^^^^^^^^^^^^^^^^^^^^^^^^^^

``TorchBackend.to_numpy`` handles all common failure modes that would
otherwise cause opaque ``RuntimeError`` exceptions:

* **Autograd-tracked tensors** (``requires_grad=True``): detached from the
  computation graph before conversion. A ``UserWarning`` is emitted.
* **Non-CPU tensors** (CUDA, MPS, …): copied to CPU host before conversion.
  A ``UserWarning`` is emitted because this incurs a device-to-host transfer.

.. code-block:: python

    import torch
    from sandalwood.backend import TorchBackend

    # Safe: detaches and warns
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    arr = TorchBackend.to_numpy(t)  # emits UserWarning

    # Safe: moves to CPU and warns
    t_gpu = torch.tensor([1.0, 2.0]).cuda()
    arr = TorchBackend.to_numpy(t_gpu)  # emits UserWarning

Behavioural fixes vs previous versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------------------------+-------------------------------+------------------------------+
| Scenario                                  | Before                        | After                        |
+===========================================+===============================+==============================+
| ``to_numpy`` on ``requires_grad`` tensor  | ``RuntimeError``              | detach + ``UserWarning``     |
+-------------------------------------------+-------------------------------+------------------------------+
| ``to_numpy`` on CUDA tensor               | ``RuntimeError``              | ``.cpu()`` + ``UserWarning`` |
+-------------------------------------------+-------------------------------+------------------------------+
| ``prod(axis=None)``                       | ``TypeError``                 | global reduction             |
+-------------------------------------------+-------------------------------+------------------------------+
| ``atleast_2d`` on 0-D tensor              | shape ``(1,)``                | shape ``(1, 1)``             |
+-------------------------------------------+-------------------------------+------------------------------+
| ``from_numpy`` default                    | zero-copy shared buffer       | independent copy             |
+-------------------------------------------+-------------------------------+------------------------------+

COSY Backend (Fortran)
~~~~~~~~~~~~~~~~~~~~~~

For maximum performance at extremely high orders (e.g., order 20+) or for specific differential algebraic operators like Poisson brackets and map composition, `sandalwood` provides a bridge to the **COSY Infinity** core.

* **High-Order Stability:** COSY is globally recognized for its numerical stability in high-order Taylor expansions.
* **Symplectic Tracking:** Optimized routines for particle tracking in accelerator physics.
* **O(1) Memory Management:** Uses a raw Fortran stack for ultra-fast allocation and deallocation.
* **Thread-safety (Python layer):** ``CosyIndexPool`` is protected by a ``threading.RLock``; ``initialize_mtf`` is protected by a module-level ``RLock``. The underlying Fortran routines are still single-threaded.

For more details on the COSY backend architecture, see :ref:`cosy_backend`.

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
