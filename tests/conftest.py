import os
import sys
import pytest
import sandalwood.taylor_function

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(params=["python", "cosy"])
def backend_implementation(request):
    """
    Parametrized fixture that yields the implementation name ('python' or 'cosy').
    Skips 'cosy' if the backend is not available.
    """
    impl = request.param
    if impl == "cosy" and not sandalwood.taylor_function._COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")
    return impl
