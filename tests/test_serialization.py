import json

import numpy as np
import pytest
from sandalwood.complex_taylor_function import ComplexMultivariateTaylorFunction
from sandalwood.taylor_function import MultivariateTaylorFunction
from sandalwood import mtf


@pytest.fixture(autouse=True)
def setup_mtf(backend_implementation):
    """Initializes MTF for serialization tests."""
    mtf.initialize_mtf(max_order=2, max_dimension=2, implementation=backend_implementation)
    # Reset initialization is handled by initialize_mtf usually, but good practice if needed


def test_json_serialization_real(backend_implementation):
    """Test serialization of a real-valued MTF."""
    mtf_obj = MultivariateTaylorFunction(
        {(1, 0): 2.0, (0, 1): 3.5}, dimension=2, var_name="f"
    )
    json_str = mtf_obj.to_json()

    # Deserialize
    mtf_loaded = MultivariateTaylorFunction.from_json(json_str)

    # Verify properties
    assert mtf_loaded.dimension == mtf_obj.dimension
    assert mtf_loaded.var_name == mtf_obj.var_name
    assert np.allclose(mtf_loaded.coeffs, mtf_obj.coeffs)
    assert np.array_equal(mtf_loaded.exponents, mtf_obj.exponents)
    assert isinstance(mtf_loaded, MultivariateTaylorFunction)
    assert not isinstance(mtf_loaded, ComplexMultivariateTaylorFunction)


def test_json_serialization_complex(backend_implementation):
    """Test serialization of a complex-valued CMTF."""
    if backend_implementation == "cosy":
        pytest.skip("COSY backend complex support is currently unstable (crashes)")

    cmtf = ComplexMultivariateTaylorFunction(
        {(1,): 2.0 + 1.5j}, dimension=1, var_name="z"
    )
    json_str = cmtf.to_json()

    # Deserialize using base class method
    cmtf_loaded = MultivariateTaylorFunction.from_json(json_str)

    # Verify properties and type promotion
    assert isinstance(cmtf_loaded, ComplexMultivariateTaylorFunction)
    assert cmtf_loaded.dimension == cmtf.dimension
    assert cmtf_loaded.var_name == cmtf.var_name
    assert np.allclose(cmtf_loaded.coeffs, cmtf.coeffs)
    assert np.array_equal(cmtf_loaded.exponents, cmtf.exponents)


def test_json_serialization_empty(backend_implementation):
    """Test serialization of an empty/zero MTF."""
    # Use dimension matches setup_mtf (max_dim=2)
    dim = 2
    mtf_obj = MultivariateTaylorFunction({}, dimension=dim)
    json_str = mtf_obj.to_json()
    mtf_loaded = MultivariateTaylorFunction.from_json(json_str)

    assert mtf_loaded.dimension == dim
    assert len(mtf_loaded.coeffs) == 0


def test_json_serialization_attributes(backend_implementation):
    """Test that serialized JSON contains expected keys."""
    mtf_obj = MultivariateTaylorFunction({(0,): 1.0}, dimension=1)
    json_str = mtf_obj.to_json()
    data = json.loads(json_str)

    expected_keys = {"dimension", "exponents", "coeffs", "is_complex", "var_name"}
    assert set(data.keys()) == expected_keys
