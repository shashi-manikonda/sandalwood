import json
import pickle

import numpy as np
import pytest
from sandalwood.complex_taylor_function import ComplexMultivariateTaylorFunction
from sandalwood.taylor_function import MultivariateTaylorFunction
from sandalwood import mtf, TaylorMap


@pytest.fixture(autouse=True)
def setup_mtf(backend_implementation):
    """Initializes MTF for serialization tests."""
    mtf.initialize_mtf(max_order=2, max_dimension=2, implementation=backend_implementation)


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
        {(1, 0): 2.0 + 1.5j}, dimension=2, var_name="z"
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
    dim = 2
    mtf_obj = MultivariateTaylorFunction({}, dimension=dim)
    json_str = mtf_obj.to_json()
    mtf_loaded = MultivariateTaylorFunction.from_json(json_str)

    assert mtf_loaded.dimension == dim
    assert len(mtf_loaded.coeffs) == 0


def test_json_serialization_attributes(backend_implementation):
    """Test that serialized JSON contains expected keys."""
    mtf_obj = MultivariateTaylorFunction({(0, 0): 1.0}, dimension=2)
    json_str = mtf_obj.to_json()
    data = json.loads(json_str)

    expected_keys = {"dimension", "exponents", "coeffs", "is_complex", "var_name"}
    assert set(data.keys()) == expected_keys


def test_json_serialization_taylor_map(backend_implementation):
    """Test serialization of a TaylorMap."""
    f1 = mtf.var(1, 2) + 0.5
    f2 = mtf.var(2, 2) ** 2
    tmap = TaylorMap([f1, f2])
    
    json_str = tmap.to_json()
    tmap_loaded = TaylorMap.from_json(json_str)
    
    assert tmap_loaded.map_dim == tmap.map_dim
    for i in range(tmap.map_dim):
        assert tmap.get_component(i) == tmap_loaded.get_component(i)


def test_pickle_roundtrip_mtf(backend_implementation):
    """Test pickle roundtrip for MTF."""
    mtf_obj = mtf.var(1, 2) + 2.5 * mtf.var(2, 2)
    pickled = pickle.dumps(mtf_obj)
    mtf_loaded = pickle.loads(pickled)
    
    assert mtf_obj == mtf_loaded
    
    if backend_implementation == "cosy":
        # Verify backend data is reconstructed
        assert mtf_loaded.mtf_data is not None
        assert np.isclose(mtf_loaded.get_constant(), mtf_obj.get_constant())


def test_pickle_roundtrip_taylor_map(backend_implementation):
    """Test pickle roundtrip for TaylorMap."""
    f1 = mtf.var(1, 2) + 1.0
    f2 = mtf.var(1, 2) * mtf.var(2, 2)
    tmap = TaylorMap([f1, f2])
    
    pickled = pickle.dumps(tmap)
    tmap_loaded = pickle.loads(pickled)
    
    assert tmap_loaded.map_dim == tmap.map_dim
    for i in range(tmap.map_dim):
        assert tmap.get_component(i) == tmap_loaded.get_component(i)
