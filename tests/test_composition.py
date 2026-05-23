import numpy as np
import pytest
import sandalwood.taylor_function as taylor

# Need to access protected members for testing low-level API before high-level integration
from sandalwood.taylor_function import MultivariateTaylorFunction


def safe_initialize(implementation):
    try:
        taylor.MultivariateTaylorFunction.initialize_mtf(
            max_order=2, max_dimension=2, implementation=implementation
        )
    except RuntimeError:
        pass


@pytest.fixture(autouse=True)
def cleanup_mtf():
    yield
    taylor.MultivariateTaylorFunction._INITIALIZED = False
    taylor.MultivariateTaylorFunction._MAX_ORDER = None
    taylor.MultivariateTaylorFunction._MAX_DIMENSION = None


@pytest.mark.parametrize("implementation", ["cosy"])
def test_compose_polval_simple(implementation):
    """Test POLVAL composition for identity and simple scaling."""
    safe_initialize(implementation)

    # f(x, y) = x
    # Ensure correct backend active

    x = MultivariateTaylorFunction.var(1)
    y = MultivariateTaylorFunction.var(2)

    if x._IMPLEMENTATION != "cosy":
        pytest.skip("Test requires COSY backend")

    # g(x, y) = 2*x
    g = 2.0 * x

    # Need full map args: [g, y]
    # Current dimension is 2.

    # Using low-level access
    f_da = x.mtf_data.da
    g_da = g.mtf_data.da
    y_da = y.mtf_data.da

    # compose_polval returns CosyDA with owned=True
    res_da = f_da.compose_polval([g_da, y_da])

    # Verify result is 2*x
    # Wrap result in MTF for easy eval
    from sandalwood.backends.cosy.cosy_backend import CosyMtfData

    # Note: CosyMtfData takes dimension. We assume it matches.
    res_data = CosyMtfData(2)
    # Manually inject da
    res_data.da = res_da  # CosyMtfData usually creates its own DA. We overwrite it.

    res_mtf = type(x)(mtf_data=res_data, dimension=2)

    val = res_mtf.eval([1, 0])[0]  # 2*1 = 2
    assert np.isclose(val, 2.0), f"Expected 2.0, got {val}"


@pytest.mark.parametrize("implementation", ["cosy"])
def test_compose_polval_product(implementation):
    safe_initialize(implementation)

    x = MultivariateTaylorFunction.var(1)
    y = MultivariateTaylorFunction.var(2)

    if x._IMPLEMENTATION != "cosy":
        pytest.skip("Test requires COSY backend")

    f = x * y  # xy

    # Substitute x -> x+1, y -> y-1
    # Result: (x+1)(y-1) = xy - x + y - 1

    sub_x = x + 1.0
    sub_y = y - 1.0

    f_da = f.mtf_data.da
    sub_x_da = sub_x.mtf_data.da
    sub_y_da = sub_y.mtf_data.da

    res_da = f_da.compose_polval([sub_x_da, sub_y_da])

    from sandalwood.backends.cosy.cosy_backend import CosyMtfData

    res_data = CosyMtfData(2)
    res_data.da = res_da
    res_mtf = type(x)(mtf_data=res_data, dimension=2)

    val_00 = res_mtf.eval([0, 0])[0]  # At 0,0: (1)(-1) = -1
    assert np.isclose(val_00, -1.0), f"Expected -1.0, got {val_00}"

    val_10 = res_mtf.eval([1, 0])[0]  # At 1,0: (2)(-1) = -2
    assert np.isclose(val_10, -2.0), f"Expected -2.0, got {val_10}"
