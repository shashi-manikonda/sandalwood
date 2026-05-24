"""
Backend abstraction for array operations.

This module provides a simple backend system to allow `sandalwood` to work with
different array libraries, such as NumPy and PyTorch. The ``get_backend``
function dynamically selects the appropriate backend **class** (not an
instance — all methods are ``@staticmethod``) based on the type of the input
array.

Design invariants
-----------------
* **Memory parity** — both backends expose an explicit ``copy`` parameter on
  ``from_numpy`` (default ``True``) so callers can opt-in to zero-copy views
  knowingly rather than having them imposed silently.
* **Device safety** — ``TorchBackend.to_numpy`` safely handles tensors that
  live on CUDA/MPS devices or carry a ``requires_grad`` flag, rather than
  raising opaque PyTorch errors.
* **Type-annotated** — every method carries a full PEP 484 signature so that
  ``mypy`` (and editors) can check call sites.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Union, overload

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
#: Union of the two supported concrete array types.
Array = Union[np.ndarray, "torch.Tensor"]
#: Shape argument accepted by zeros / ones.
Shape = Union[int, tuple[int, ...]]
#: Optional dtype accepted by zeros / ones.
DType = Optional[Union[np.dtype, "torch.dtype"]]


# ---------------------------------------------------------------------------
# NumpyBackend
# ---------------------------------------------------------------------------


class NumpyBackend:
    """
    A backend that uses NumPy for array operations.

    This class provides a set of static methods that wrap common NumPy
    functions, conforming to the interface expected by the ``neval`` method
    and other parts of ``sandalwood``.

    All methods are ``@staticmethod``; callers should prefer
    ``get_backend(array)`` over instantiating this class directly.
    """

    @staticmethod
    def power(base: np.ndarray, exp: np.ndarray) -> np.ndarray:
        """Wraps ``np.power``."""
        return np.power(base, exp)

    @staticmethod
    def prod(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Wraps ``np.prod``.

        Parameters
        ----------
        a:
            Input array.
        axis:
            Axis along which to compute the product. ``None`` (default)
            reduces over all elements, matching ``np.prod`` semantics.
        """
        return np.prod(a, axis=axis)

    @staticmethod
    def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Wraps ``np.dot``."""
        return np.dot(a, b)

    @staticmethod
    def zeros(shape: Shape, dtype: DType = None) -> np.ndarray:
        """Wraps ``np.zeros``."""
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape: Shape, dtype: DType = None) -> np.ndarray:
        """Wraps ``np.ones``."""
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def atleast_2d(a: np.ndarray) -> np.ndarray:
        """Wraps ``np.atleast_2d``.

        Handles 0-D, 1-D, and higher-dimensional arrays correctly.
        """
        return np.atleast_2d(a)

    @staticmethod
    def from_numpy(a: np.ndarray, copy: bool = True) -> np.ndarray:
        """Converts a NumPy-compatible array to a NumPy array.

        Parameters
        ----------
        a:
            Source array.
        copy:
            If ``True`` (default), always returns a new copy of the data.
            If ``False``, returns a view if the array is already a
            compatible contiguous ``float64``/``complex128`` array, which
            avoids an allocation but means mutations on the returned array
            will propagate back to the source.
        """
        if copy:
            return np.array(a)
        return np.asarray(a)

    @staticmethod
    def to_numpy(a: np.ndarray) -> np.ndarray:
        """Converts a NumPy array to a standard NumPy array (identity).

        Returns a view of the input when possible (``np.asarray`` semantics).
        """
        return np.asarray(a)


# ---------------------------------------------------------------------------
# TorchBackend (conditionally defined)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class TorchBackend:
        """
        A backend that uses PyTorch for array operations.

        This class provides a set of static methods that wrap common PyTorch
        functions, conforming to the interface expected by the ``neval``
        method. It is only available if PyTorch is installed.

        Device & autograd safety
        ------------------------
        ``to_numpy`` handles all common failure modes (CUDA/MPS tensors and
        autograd-tracked tensors) instead of raising cryptic PyTorch errors.
        See its docstring for details.

        Memory semantics
        ----------------
        ``from_numpy(a, copy=True)`` (the default) returns a clone of the
        data — unlike the bare ``torch.from_numpy`` which would share the
        underlying buffer. Pass ``copy=False`` only when you know you need a
        zero-copy view and will not mutate the returned tensor.
        """

        @staticmethod
        def power(base: "torch.Tensor", exp: "torch.Tensor") -> "torch.Tensor":
            """Wraps ``torch.pow``."""
            return torch.pow(base, exp)

        @staticmethod
        def prod(a: "torch.Tensor", axis: Optional[int] = None) -> "torch.Tensor":
            """Wraps ``torch.prod`` with correct ``axis=None`` handling.

            Parameters
            ----------
            a:
                Input tensor.
            axis:
                Dimension along which to reduce. When ``None`` (default),
                reduces over **all** elements to a scalar tensor — matching
                ``np.prod(a, axis=None)`` semantics. Passing ``None`` directly
                to ``torch.prod(dim=None)`` would raise a ``TypeError``; this
                wrapper handles that case correctly.
            """
            if axis is None:
                # Global reduction — no dim argument
                return torch.prod(a)
            return torch.prod(a, dim=axis)

        @staticmethod
        def dot(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
            """Wraps ``torch.matmul`` for dot product with type handling.

            Promotes the lower-precision operand to complex if one operand
            is complex and the other is not, mirroring NumPy's casting rules.
            """
            if a.dtype != b.dtype:
                if b.is_complex() and not a.is_complex():
                    a = a.to(b.dtype)
                elif a.is_complex() and not b.is_complex():
                    b = b.to(a.dtype)
            return torch.matmul(a, b)

        @staticmethod
        def zeros(shape: Shape, dtype: DType = None) -> "torch.Tensor":
            """Wraps ``torch.zeros``."""
            return torch.zeros(shape, dtype=dtype)

        @staticmethod
        def ones(shape: Shape, dtype: DType = None) -> "torch.Tensor":
            """Wraps ``torch.ones``."""
            return torch.ones(shape, dtype=dtype)

        @staticmethod
        def atleast_2d(a: "torch.Tensor") -> "torch.Tensor":
            """Ensures the tensor is at least 2-D.

            Mirrors ``np.atleast_2d`` for 0-D, 1-D, and higher-dimensional
            tensors:

            * **0-D** scalar tensor  → shape ``(1, 1)``
            * **1-D** tensor of shape ``(N,)`` → shape ``(1, N)``
            * **≥2-D** tensors are returned unchanged.
            """
            if a.dim() == 0:
                return a.unsqueeze(0).unsqueeze(0)  # (1, 1)
            if a.dim() == 1:
                return a.unsqueeze(0)  # (1, N)
            return a

        @staticmethod
        def from_numpy(a: np.ndarray, copy: bool = True) -> "torch.Tensor":
            """Wraps ``torch.from_numpy`` with explicit copy control.

            Parameters
            ----------
            a:
                Source NumPy array. Must be CPU-resident and of a dtype
                supported by ``torch.from_numpy`` (e.g. ``float64``,
                ``complex128``, ``int32``).
            copy:
                If ``True`` (default), returns an independent clone of the
                data.  The returned tensor does **not** share memory with
                ``a`` — subsequent mutations of either ``a`` or the tensor
                will not affect the other.

                If ``False``, returns a zero-copy view via
                ``torch.from_numpy``.  In this mode the tensor and ``a``
                share the same memory buffer; in-place operations on either
                will be visible in the other.  Use only when you are certain
                no mutation will occur.

            Raises
            ------
            TypeError
                If ``a`` is not a NumPy ndarray or has an unsupported dtype.
            """
            t = torch.from_numpy(a)
            if copy:
                return t.clone()
            return t

        @staticmethod
        def to_numpy(a: "torch.Tensor") -> np.ndarray:
            """Safely converts a PyTorch tensor to a NumPy array.

            Handles all common failure modes that would otherwise cause opaque
            PyTorch ``RuntimeError`` exceptions:

            * **Autograd-tracked tensors** (``requires_grad=True``) — the
              tensor is detached from the computation graph before conversion.
              A ``UserWarning`` is emitted so callers are aware that gradient
              information is not preserved.
            * **Non-CPU tensors** (CUDA, MPS, …) — the tensor is copied to
              the CPU host before conversion.  A ``UserWarning`` is emitted
              because this incurs a device-to-host transfer that may be
              expensive and surprising.

            Parameters
            ----------
            a:
                The tensor to convert.

            Returns
            -------
            np.ndarray
                A CPU NumPy array.  The result may share memory with ``a``
                when ``a`` was already a detached, contiguous CPU tensor
                (``torch.Tensor.numpy()`` returns a view in that case).

            Warns
            -----
            UserWarning
                If ``a.requires_grad`` is ``True`` (gradient detached).
            UserWarning
                If ``a`` lives on a non-CPU device (host transfer performed).
            """
            # Step 1 — detach from autograd graph if necessary
            if a.requires_grad:
                warnings.warn(
                    "TorchBackend.to_numpy(): tensor has requires_grad=True. "
                    "Detaching from the computation graph. Gradients will NOT "
                    "be propagated through this conversion.",
                    UserWarning,
                    stacklevel=2,
                )
                a = a.detach()

            # Step 2 — move to CPU if necessary
            if a.device.type != "cpu":
                warnings.warn(
                    f"TorchBackend.to_numpy(): tensor is on device "
                    f"'{a.device}'. Copying to CPU for NumPy conversion. "
                    "This incurs a device-to-host memory transfer.",
                    UserWarning,
                    stacklevel=2,
                )
                a = a.cpu()

            # Step 3 — convert (safe: detached, contiguous, CPU)
            return a.numpy()


# ---------------------------------------------------------------------------
# Backend registry and get_backend dispatcher
# ---------------------------------------------------------------------------

# Singletons — all methods are @staticmethod; no instance state exists.
# Returning the class avoids constructing a fresh instance on every dispatch.
_backends: dict[type, type] = {
    np.ndarray: NumpyBackend,
}
if _TORCH_AVAILABLE:
    _backends[torch.Tensor] = TorchBackend


@overload
def get_backend(array: np.ndarray) -> type[NumpyBackend]: ...  # type: ignore[overload-overlap]


if TYPE_CHECKING:

    @overload
    def get_backend(array: "torch.Tensor") -> "type[TorchBackend]": ...


def get_backend(array: Array) -> "type[NumpyBackend] | type[TorchBackend]":
    """
    Selects and returns the appropriate backend **class** for a given array.

    Because all backend methods are ``@staticmethod``, the class itself is
    the callable — no instance construction is required, avoiding per-call
    allocation overhead.

    Parameters
    ----------
    array:
        The array for which to find a backend. Supported types are
        ``np.ndarray`` and (when PyTorch is installed) ``torch.Tensor``.
        Subclasses of either type are also supported via ``isinstance``
        fallback.

    Returns
    -------
    type[NumpyBackend] | type[TorchBackend]
        The backend class appropriate for the input array type.

    Raises
    ------
    TypeError
        If the array type is not supported.
    """
    # Fast path: exact type match (O(1) dictionary lookup)
    backend_cls = _backends.get(type(array))
    if backend_cls is not None:
        return backend_cls

    # Fallback: isinstance check for ndarray/Tensor subclasses
    if isinstance(array, np.ndarray):
        return NumpyBackend
    if _TORCH_AVAILABLE and isinstance(array, torch.Tensor):
        return TorchBackend

    raise TypeError(
        f"Unsupported array type: {type(array).__qualname__}. "
        f"Supported types: {[t.__name__ for t in _backends]}"
    )
