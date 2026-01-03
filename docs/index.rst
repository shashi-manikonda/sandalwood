Welcome to mtflib's documentation!
==================================

`mtflib` is a Python library for creating, manipulating, and composing
Multivariate Taylor Functions, with a C++ backend for performance-critical
applications. It provides a robust framework for working with multivariate
Taylor series expansions based on the principles of Differential Algebra (DA).

This library is designed for scientists, engineers, and researchers who need
to perform high-order differentiation, integration, and function composition
in a computationally efficient manner.

#### Features
* **High Performance:** `mtflib` utilizes a C++ backend for key operations, ensuring fast, efficient computation.
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

.. toctree::
   :maxdepth: 1
   :caption: Project Info

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
