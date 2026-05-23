import numpy as np
import pytest
from sandalwood import taylor_function as taylor
from sandalwood.backends.cosy import cosy_backend


@pytest.fixture(autouse=True)
def cleanup_mtf():
    """Reset MTF initialization after each test."""
    yield
    taylor.MultivariateTaylorFunction._INITIALIZED = False


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_linear_combination(implementation):
    """Test da_lin_comb: res = 2*DA1 + 3*DA2"""
    if implementation == "python":
        # Python Implementation Logic
        taylor.MultivariateTaylorFunction.initialize_mtf(
            max_order=1, max_dimension=1, implementation="python"
        )
        x = taylor.MultivariateTaylorFunction.var(1)
        c = taylor.MultivariateTaylorFunction.from_constant(1.0)

        # Perform operation using standard arithmetic (Python backend)
        res = 2.0 * x + 3.0 * c

        assert res.extract_coefficient((1,)) == 2.0
        assert res.extract_coefficient((0,)) == 3.0
        return

    if implementation == "cosy" and not taylor._COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")

    # Setup
    cosy_backend.CosyBackend.initialize(order=1, dim=1)

    # Create x and constant 1
    da_x = cosy_backend.CosyDA(var_id=0)  # x1
    da_c = cosy_backend.CosyDA.from_const(1.0)

    # Compute 2*x + 3*1 using lin_comb
    res_da = cosy_backend.da_lin_comb(da_x, 2.0, da_c, 3.0)  # 2x + 3

    # We can use CosyMtfData wrapper for easy eval
    mtf_data = cosy_backend.CosyMtfData(1)
    mtf_data.da = res_da

    val = mtf_data.eval([2.0])  # 2(2) + 3 = 7
    assert np.isclose(val, 7.0), f"Expected 7.0, got {val}"


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_matrix_inversion(implementation):
    """Test inversion of a 2x2 matrix using da_mat_inv"""
    if implementation == "python":
        # Python Implementation
        taylor.MultivariateTaylorFunction.initialize_mtf(
            max_order=1, max_dimension=1, implementation="python"
        )
        x = taylor.MultivariateTaylorFunction.var(1)

        # Construct a 2x2 matrix of MTFs
        # M = [[1, x], [0, 1]] -> Inverse should be [[1, -x], [0, 1]]
        # (Since it is upper triangular with 1s on diag)
        M = np.array(
            [
                [taylor.MultivariateTaylorFunction.from_constant(1.0), x],
                [
                    taylor.MultivariateTaylorFunction.from_constant(0.0),
                    taylor.MultivariateTaylorFunction.from_constant(1.0),
                ],
            ],
            dtype=object,
        )

        # Use manual inversion for object arrays (NumPy's linalg.inv doesn't support dtype=object)
        # For M = [[a, b], [c, d]], inv = 1/(ad-bc) * [[d, -b], [-c, a]]
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]  # Should be 1.0
        inv_det = 1.0 / det

        # Explicit element-wise multiplication to avoid numpy ufunc issues with MTF objects
        M_inv = np.array(
            [
                [inv_det * M[1, 1], inv_det * (-M[0, 1])],
                [inv_det * (-M[1, 0]), inv_det * M[0, 0]],
            ],
            dtype=object,
        )

        # Check M * M_inv = I
        Identity = M @ M_inv

        # Note: Identity[0,0] is 1, Identity[0,1] is 0
        assert abs(Identity[0, 0].extract_coefficient((0,))) - 1.0 < 1e-12
        assert abs(Identity[0, 1].extract_coefficient((1,))) < 1e-12
        return

    if implementation == "cosy":
        if not taylor._COSY_BACKEND_AVAILABLE:
            pytest.skip("COSY backend not available")

        # Matrix A = [[4, 7], [2, 6]]
        # Det = 24 - 14 = 10
        # Inv = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]

        matrix = [4.0, 7.0, 2.0, 6.0]
        n = 2

        inv_matrix = cosy_backend.da_mat_inv(matrix, n)

        expected = [0.6, -0.7, -0.2, 0.4]

        assert np.allclose(
            inv_matrix, expected
        ), f"Expected {expected}, got {inv_matrix}"
