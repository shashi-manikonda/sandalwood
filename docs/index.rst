Welcome to sandalwood's documentation!
==================================

`sandalwood` is a Python library for creating, manipulating, and composing
Multivariate Taylor Functions, with high-performance backends for 
acceleration. It provides a robust framework for working with multivariate
Taylor series expansions based on the principles of Differential Algebra (DA).

This library is designed for scientists, engineers, and researchers who need
to perform high-order differentiation, integration, and function composition
in a computationally efficient manner.

#### Features
* **Numba-Optimized Python Backend:** Utilizes JIT compilation and Dense Mode architecture for order-of-magnitude speedups in native Python.
* **COSY Backend (Fortran):** Leverages the battle-tested COSY Infinity core for extremely high-order calculations and symplectic tracking.
* **Backend Flexibility:** Supports both NumPy and PyTorch, automatically switching backends to leverage GPU acceleration when PyTorch tensors are used.
* **Comprehensive Functionality:** Includes a wide range of elementary functions and core operations like composition, differentiation, and integration.


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   examples
   taylor_map

.. toctree::
   :maxdepth: 2
   :caption: API

   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Background

   mtf_background

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced_topics
   optimization
   cosy_backend
   benchmarking

.. toctree::
   :maxdepth: 1
   :caption: Project Info

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
