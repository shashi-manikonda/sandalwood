"""
Unit tests for sandalwood.backend
==================================

These tests directly exercise the backend abstraction layer introduced (and
rewritten) in feat/backend-hardening.  They are intentionally independent of
``MultivariateTaylorFunction`` so they can run without any MTF initialisation
and stay fast.

Torch-specific tests are unconditionally skipped when PyTorch is not installed.
CUDA tests are additionally skipped when no CUDA device is available.
"""

import warnings

import numpy as np
import pytest

from sandalwood.backend import NumpyBackend, _TORCH_AVAILABLE, get_backend

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:
    import torch
    from sandalwood.backend import TorchBackend

_skip_no_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch not installed"
)
_skip_no_cuda = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and __import__("torch").cuda.is_available()),
    reason="CUDA not available",
)


# ===========================================================================
# get_backend — dispatcher
# ===========================================================================


class TestGetBackend:
    """Tests for the get_backend() dispatcher function."""

    def test_returns_class_not_instance_numpy(self):
        """get_backend must return the class itself, not an instance."""
        arr = np.array([1.0, 2.0])
        result = get_backend(arr)
        assert result is NumpyBackend, "Expected NumpyBackend class, got instance or wrong type"
        assert not isinstance(result, NumpyBackend), (
            "get_backend should return the class (type), not an instance"
        )

    def test_numpy_array_dispatches_to_numpy_backend(self):
        """np.ndarray → NumpyBackend."""
        arr = np.zeros((3, 4), dtype=np.float64)
        assert get_backend(arr) is NumpyBackend

    def test_numpy_subclass_dispatches_to_numpy_backend(self):
        """Subclasses of np.ndarray should also dispatch to NumpyBackend."""

        class MyArray(np.ndarray):
            pass

        arr = np.array([1.0]).view(MyArray)
        assert get_backend(arr) is NumpyBackend

    @_skip_no_torch
    def test_torch_tensor_dispatches_to_torch_backend(self):
        """torch.Tensor → TorchBackend."""
        t = torch.tensor([1.0, 2.0])
        assert get_backend(t) is TorchBackend

    @_skip_no_torch
    def test_returns_class_not_instance_torch(self):
        """get_backend must return TorchBackend class, not an instance."""
        t = torch.tensor([1.0])
        result = get_backend(t)
        assert result is TorchBackend
        assert not isinstance(result, TorchBackend)

    def test_unsupported_type_raises_type_error(self):
        """Non-array types must raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported array type"):
            get_backend("a string")

    def test_unsupported_type_list_raises_type_error(self):
        """Plain Python list must raise TypeError."""
        with pytest.raises(TypeError):
            get_backend([1, 2, 3])


# ===========================================================================
# NumpyBackend
# ===========================================================================


class TestNumpyBackendFromNumpy:
    """Tests for NumpyBackend.from_numpy copy semantics."""

    def test_copy_true_is_independent(self):
        """copy=True (default) — mutating the result must NOT affect the source."""
        src = np.array([1.0, 2.0, 3.0])
        result = NumpyBackend.from_numpy(src, copy=True)
        result[0] = 999.0
        assert src[0] == 1.0, "Source array was mutated; copy=True did not copy"

    def test_copy_false_shares_buffer(self):
        """copy=False — result shares memory with source (mutation propagates)."""
        src = np.array([1.0, 2.0, 3.0])
        result = NumpyBackend.from_numpy(src, copy=False)
        result[0] = 999.0
        assert src[0] == 999.0, "Source not updated; copy=False should share buffer"

    def test_default_copy_is_true(self):
        """Default parameter must behave as copy=True."""
        src = np.array([1.0, 2.0])
        result = NumpyBackend.from_numpy(src)
        result[1] = -99.0
        assert src[1] == 2.0


class TestNumpyBackendProd:
    """Tests for NumpyBackend.prod axis handling."""

    def test_prod_axis_none_global_reduction(self):
        """axis=None must reduce over all elements."""
        a = np.array([2.0, 3.0, 4.0])
        result = NumpyBackend.prod(a, axis=None)
        assert np.isclose(result, 24.0)

    def test_prod_axis_0(self):
        """axis=0 reduces along rows of a 2-D array."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = NumpyBackend.prod(a, axis=0)
        np.testing.assert_allclose(result, [3.0, 8.0])

    def test_prod_default_axis_is_none(self):
        """Default call (no axis) is equivalent to axis=None."""
        a = np.array([2.0, 5.0])
        assert np.isclose(NumpyBackend.prod(a), 10.0)


class TestNumpyBackendAtleast2d:
    """Tests for NumpyBackend.atleast_2d shape handling."""

    def test_0d_scalar_becomes_1x1(self):
        """0-D array → shape (1, 1)."""
        a = np.array(5.0)
        result = NumpyBackend.atleast_2d(a)
        assert result.shape == (1, 1)

    def test_1d_becomes_1xN(self):
        """1-D array of length N → shape (1, N)."""
        a = np.array([1.0, 2.0, 3.0])
        result = NumpyBackend.atleast_2d(a)
        assert result.shape == (1, 3)

    def test_2d_unchanged(self):
        """2-D array is returned unchanged."""
        a = np.zeros((4, 5))
        result = NumpyBackend.atleast_2d(a)
        assert result.shape == (4, 5)


# ===========================================================================
# TorchBackend
# ===========================================================================


@_skip_no_torch
class TestTorchBackendFromNumpy:
    """Tests for TorchBackend.from_numpy copy semantics."""

    def test_copy_true_is_independent(self):
        """copy=True (default) — mutating the tensor must NOT affect the NumPy source."""
        src = np.array([1.0, 2.0, 3.0])
        t = TorchBackend.from_numpy(src, copy=True)
        t[0] = 999.0
        assert src[0] == 1.0, "Source NumPy array was mutated; clone failed"

    def test_copy_false_shares_buffer(self):
        """copy=False — tensor and source share memory."""
        src = np.array([1.0, 2.0, 3.0])
        t = TorchBackend.from_numpy(src, copy=False)
        t[0] = 999.0
        assert src[0] == 999.0, "Source not updated; zero-copy view expected"

    def test_default_copy_is_true(self):
        """Default parameter must behave as copy=True (safe clone)."""
        src = np.array([4.0, 5.0])
        t = TorchBackend.from_numpy(src)
        t[0] = -1.0
        assert src[0] == 4.0


@_skip_no_torch
class TestTorchBackendToNumpy:
    """Tests for TorchBackend.to_numpy safety guarantees."""

    def test_cpu_nograd_tensor_no_warning(self):
        """Clean CPU tensor with no grad: converts without warning."""
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → failure
            arr = TorchBackend.to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_requires_grad_emits_user_warning(self):
        """requires_grad=True: must emit UserWarning and still return ndarray."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        with pytest.warns(UserWarning, match="requires_grad"):
            arr = TorchBackend.to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0])

    def test_requires_grad_gradient_is_not_preserved(self):
        """After to_numpy on a grad tensor the result is a plain ndarray (no grad)."""
        t = torch.tensor([3.0, 4.0], requires_grad=True)
        with pytest.warns(UserWarning):
            arr = TorchBackend.to_numpy(t)
        # arr is a plain numpy array — no grad attribute
        assert not hasattr(arr, "requires_grad")

    @pytest.mark.skipif(
        not (_TORCH_AVAILABLE and __import__("torch").cuda.is_available()),
        reason="CUDA not available",
    )
    def test_cuda_tensor_emits_user_warning(self):
        """CUDA tensor: must emit UserWarning and return CPU ndarray."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float64).cuda()
        with pytest.warns(UserWarning, match="device"):
            arr = TorchBackend.to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0])


@_skip_no_torch
class TestTorchBackendProd:
    """Tests for TorchBackend.prod — guards the axis=None TypeError fix."""

    def test_prod_axis_none_global_reduction(self):
        """axis=None must perform a global reduction — was TypeError before fix."""
        t = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        result = TorchBackend.prod(t, axis=None)
        assert torch.isclose(result, torch.tensor(24.0, dtype=torch.float64))

    def test_prod_default_is_global(self):
        """Calling prod() with no axis argument must also work (default=None)."""
        t = torch.tensor([2.0, 5.0], dtype=torch.float64)
        result = TorchBackend.prod(t)
        assert torch.isclose(result, torch.tensor(10.0, dtype=torch.float64))

    def test_prod_axis_0(self):
        """axis=0 reduces along rows of a 2-D tensor."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        result = TorchBackend.prod(t, axis=0)
        np.testing.assert_allclose(result.numpy(), [3.0, 8.0])


@_skip_no_torch
class TestTorchBackendAtleast2d:
    """Tests for TorchBackend.atleast_2d — guards the 0-D shape fix."""

    def test_0d_scalar_becomes_1x1(self):
        """0-D scalar tensor → shape (1, 1).  Was (1,) before fix."""
        t = torch.tensor(5.0)
        result = TorchBackend.atleast_2d(t)
        assert result.shape == torch.Size([1, 1])

    def test_1d_becomes_1xN(self):
        """1-D tensor of length N → shape (1, N)."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = TorchBackend.atleast_2d(t)
        assert result.shape == torch.Size([1, 3])

    def test_2d_unchanged(self):
        """2-D tensor is returned unchanged."""
        t = torch.zeros(4, 5)
        result = TorchBackend.atleast_2d(t)
        assert result.shape == torch.Size([4, 5])


@_skip_no_torch
class TestTorchBackendDot:
    """Tests for TorchBackend.dot dtype promotion."""

    def test_real_real_dot(self):
        """Standard real dot product."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        result = TorchBackend.dot(a, b)
        assert torch.isclose(result, torch.tensor(32.0, dtype=torch.float64))

    def test_complex_promotion_b_is_complex(self):
        """real @ complex: a should be promoted to complex dtype automatically."""
        a = torch.tensor([1.0, 0.0], dtype=torch.float64)
        b = torch.tensor([1.0 + 1j, 0.0 + 1j], dtype=torch.complex128)
        result = TorchBackend.dot(a, b)
        assert result.is_complex()
        # 1*(1+1j) + 0*(0+1j) = 1+1j
        assert torch.isclose(result, torch.tensor(1.0 + 1.0j, dtype=torch.complex128))

    def test_complex_promotion_a_is_complex(self):
        """complex @ real: b should be promoted to complex dtype automatically."""
        a = torch.tensor([1.0 + 1j, 0.0], dtype=torch.complex128)
        b = torch.tensor([2.0, 3.0], dtype=torch.float64)
        result = TorchBackend.dot(a, b)
        assert result.is_complex()
        # (1+1j)*2 + 0*3 = 2+2j
        assert torch.isclose(result, torch.tensor(2.0 + 2.0j, dtype=torch.complex128))
