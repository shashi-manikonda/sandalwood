"""
Backend abstraction for array operations.

This module provides a simple backend system to allow `mtflib` to work with
different array libraries, such as NumPy and PyTorch. The `get_backend`
function dynamically selects the appropriate backend based on the type of
the input array. Each backend class (`NumpyBackend`, `TorchBackend`) wraps
the corresponding library's functions in a common interface.
"""

import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass


class NumpyBackend:
    """
    A backend that uses NumPy for array operations.

    This class provides a set of static methods that wrap common NumPy
    functions, conforming to the interface expected by the `neval` method
    and other parts of `mtflib`.
    """

    @staticmethod
    def power(base, exp):
        """Wraps `np.power`."""
        return np.power(base, exp)

    @staticmethod
    def prod(a, axis=None):
        """Wraps `np.prod`."""
        return np.prod(a, axis=axis)

    @staticmethod
    def dot(a, b):
        """Wraps `np.dot`."""
        return np.dot(a, b)

    @staticmethod
    def zeros(shape, dtype=None):
        """Wraps `np.zeros`."""
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def atleast_2d(a):
        """Wraps `np.atleast_2d`."""
        return np.atleast_2d(a)

    @staticmethod
    def from_numpy(a):
        """Converts a NumPy-compatible array to a NumPy array."""
        return np.array(a)

    @staticmethod
    def to_numpy(a):
        """Converts a NumPy array to a standard NumPy array (identity)."""
        return np.array(a)


if _TORCH_AVAILABLE:

    class TorchBackend:
        """
        A backend that uses PyTorch for array operations.

        This class provides a set of static methods that wrap common PyTorch
        functions, conforming to the interface expected by the `neval` method.
        It is only available if PyTorch is installed.
        """

        @staticmethod
        def power(base, exp):
            """Wraps `torch.pow`."""
            return torch.pow(base, exp)

        @staticmethod
        def prod(a, axis=None):
            """Wraps `torch.prod`."""
            return torch.prod(a, dim=axis)

        @staticmethod
        def dot(a, b):
            """Wraps `torch.matmul` for dot product."""
            return torch.matmul(a, b)

        @staticmethod
        def zeros(shape, dtype=None):
            """Wraps `torch.zeros`."""
            return torch.zeros(shape, dtype=dtype)

        @staticmethod
        def atleast_2d(a):
            """Ensures the tensor is at least 2D."""
            if a.dim() >= 2:
                return a
            return a.unsqueeze(0)

        @staticmethod
        def from_numpy(a):
            """Wraps `torch.from_numpy`."""
            return torch.from_numpy(a)

        @staticmethod
        def to_numpy(a):
            """Converts a PyTorch tensor to a NumPy array."""
            return a.numpy()


_backends = {
    np.ndarray: NumpyBackend,
}
if _TORCH_AVAILABLE:
    _backends[torch.Tensor] = TorchBackend


def get_backend(array):
    """
    Selects and returns the appropriate backend for a given array object.

    This function inspects the type of the input array and returns an
    instance of the corresponding backend class (`NumpyBackend` or
    `TorchBackend`).

    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        The array for which to find a backend.

    Returns
    -------
    NumpyBackend or TorchBackend
        An instance of the appropriate backend class.

    Raises
    ------
    TypeError
        If the array type is not supported.
    """
    array_type = type(array)
    if array_type in _backends:
        return _backends[array_type]()

    # Fallback for subclasses
    if isinstance(array, np.ndarray):
        return NumpyBackend()
    if _TORCH_AVAILABLE and isinstance(array, torch.Tensor):
        return TorchBackend()

    raise TypeError(f"Unsupported array type: {array_type}")
