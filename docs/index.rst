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
* **Hybrid Multicore Architecture:** Seamlessly combines JIT-compiled Python (Numba), vectorized Fortran (COSY Infinity), and GPU-accelerated backends.
* **Robust Memory Management:** Features the `CosyIndexPool` for O(1) memory allocation and strict `CosyScope` tracking for complex simulations.
* **Massive Order Support:** Efficiently handles Taylor expansions up to order 128+ with specialized dense-mode multiplication tables.
* **High-Performance Physics:** Includes optimized kernels for Biot-Savart integration and high-order symplectic Map Composition.


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
