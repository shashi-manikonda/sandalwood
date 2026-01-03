import numpy as np
import pytest

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
from mtflib import mtf


# Old eval logic for comparison
def old_eval(mtf, evaluation_point):
    """The old eval logic for comparison."""
    evaluation_point = np.array(evaluation_point)
    term_values = np.prod(np.power(evaluation_point, mtf.exponents), axis=1)
    result = np.einsum("j,j->", mtf.coeffs, term_values)
    return result


@pytest.fixture(autouse=True)
def reset_mtf():
    """Resets the MTF library before each test."""
    mtf._INITIALIZED = False


@pytest.fixture
def sample_mtf():
    """A sample MTF for testing."""
    mtf.initialize_mtf(max_order=2, max_dimension=2)
    x = mtf.var(1, 2)
    y = mtf.var(2, 2)
    return x**2 + 2 * x * y + y**2 + 1


def test_neval_single_point(sample_mtf):
    """Tests neval with a single point and compares with old eval."""
    point = np.array([1.0, 2.0])

    # Expected result from old eval
    expected = old_eval(sample_mtf, point)

    # Result from neval
    result = sample_mtf.neval(point.reshape(1, -1))

    assert result.shape == (1,)
    assert np.allclose(result[0], expected)


def test_neval_multiple_points(sample_mtf):
    """Tests neval with multiple points."""
    points = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])

    expected = np.array([
        old_eval(sample_mtf, points[0]),
        old_eval(sample_mtf, points[1]),
        old_eval(sample_mtf, points[2]),
    ])

    result = sample_mtf.neval(points)

    assert result.shape == (3,)
    assert np.allclose(result, expected)


def test_neval_error_handling(sample_mtf):
    """Tests the error handling of neval."""
    with pytest.raises(ValueError):
        # Incorrect dimension
        sample_mtf.neval(np.array([[1.0]]))

    with pytest.raises(ValueError):
        # Incorrect shape
        sample_mtf.neval(np.array([1.0, 2.0, 3.0]))


def test_eval_wrapper(sample_mtf):
    """Tests the new eval wrapper."""
    point = np.array([1.0, 2.0])

    # Expected result from old eval
    expected = old_eval(sample_mtf, point)

    # Result from new eval
    result = sample_mtf.eval(point)

    assert np.allclose(result, expected)

    # Test with 2D input
    result_2d = sample_mtf.eval(point.reshape(1, -1))
    assert np.allclose(result_2d, expected)

    # Test error handling for 2D input with more than one point
    with pytest.raises(ValueError):
        sample_mtf.eval(np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_neval_torch_tensor(sample_mtf):
    """Tests neval with a torch tensor."""
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)

    expected = np.array([
        old_eval(sample_mtf, points[0].numpy()),
        old_eval(sample_mtf, points[1].numpy()),
        old_eval(sample_mtf, points[2].numpy()),
    ])

    result = sample_mtf.neval(points)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)
    assert np.allclose(result.numpy(), expected)
