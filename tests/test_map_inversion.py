from sandalwood import TaylorMap, mtf


import pytest
import sandalwood.taylor_function

def test_map_inversion_from_demo(backend_implementation):
    """
    This test replicates the map inversion demo from taylor_map_demo.ipynb
    to verify if it's broken after recent changes.
    """
    implementation = backend_implementation

    # 1. Initialize sandalwood
    mtf._INITIALIZED = False
    mtf.initialize_mtf(max_order=4, max_dimension=2, implementation=implementation)

    # 2. Create the invertible map
    x = mtf.var(1, 2)
    y = mtf.var(2, 2)
    f1 = x + 0.1 * y**2
    f2 = y - 0.1 * x**2
    map_to_invert = TaylorMap([f1, f2])

    # 3. Invert the map
    inverted_map = map_to_invert.invert()

    # 4. Verify by composing F and F_inv
    composition = inverted_map.compose(map_to_invert)

    # 5. The result should be the identity map [x, y]
    identity_map = TaylorMap([x, y])

    # 6. Check that the components are equal
    # The __eq__ method for TaylorMap components handles necessary cleanup and
    # tolerance, so we can compare them directly without truncation.
    # This provides a stricter test that would have caught the regression.
    assert composition.get_component(0) == identity_map.get_component(0)
    assert composition.get_component(1) == identity_map.get_component(1)
