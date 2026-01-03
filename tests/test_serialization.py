import json

import numpy as np
from mtflib.complex_taylor_function import ComplexMultivariateTaylorFunction
from mtflib.taylor_function import MultivariateTaylorFunction


def test_json_serialization_real():
    """Test serialization of a real-valued MTF."""
    mtf = MultivariateTaylorFunction(
        {(1, 0): 2.0, (0, 1): 3.5}, dimension=2, var_name="f"
    )
    json_str = mtf.to_json()

    # Deserialize
    mtf_loaded = MultivariateTaylorFunction.from_json(json_str)

    # Verify properties
    assert mtf_loaded.dimension == mtf.dimension
    assert mtf_loaded.var_name == mtf.var_name
    assert np.allclose(mtf_loaded.coeffs, mtf.coeffs)
    assert np.array_equal(mtf_loaded.exponents, mtf.exponents)
    assert isinstance(mtf_loaded, MultivariateTaylorFunction)
    assert not isinstance(mtf_loaded, ComplexMultivariateTaylorFunction)


def test_json_serialization_complex():
    """Test serialization of a complex-valued CMTF."""
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


def test_json_serialization_empty():
    """Test serialization of an empty/zero MTF."""
    mtf = MultivariateTaylorFunction({}, dimension=3)
    json_str = mtf.to_json()
    mtf_loaded = MultivariateTaylorFunction.from_json(json_str)

    assert mtf_loaded.dimension == 3
    assert len(mtf_loaded.coeffs) == 0


def test_json_serialization_attributes():
    """Test that serialized JSON contains expected keys."""
    mtf = MultivariateTaylorFunction({(0,): 1.0}, dimension=1)
    json_str = mtf.to_json()
    data = json.loads(json_str)

    expected_keys = {"dimension", "exponents", "coeffs", "is_complex", "var_name"}
    assert set(data.keys()) == expected_keys
