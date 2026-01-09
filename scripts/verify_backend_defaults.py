from sandalwood.taylor_function import _COSY_BACKEND_AVAILABLE
from sandalwood.taylor_function import MultivariateTaylorFunction as MTF


def test_default_backend():
    print("Testing default backend initialization...")
    print(f"COSY Available: {_COSY_BACKEND_AVAILABLE}")

    # Reset initialization
    MTF._INITIALIZED = False
    MTF.initialize_mtf(max_order=2, max_dimension=2)

    print(f"Implementation chosen: {MTF._IMPLEMENTATION}")

    if _COSY_BACKEND_AVAILABLE:
        assert MTF._IMPLEMENTATION == "cosy", "Should default to COSY when available"
    else:
        assert MTF._IMPLEMENTATION == "python", (
            "Should fallback to Python when COSY unavailable"
        )

    print("Default backend test passed.")


def test_fallback_simulation():
    print("\nTesting fallback logic (Simulated)...")
    # Simulate COSY unavailable locally for this test logic
    # We can't easily change _COSY_BACKEND_AVAILABLE module-level variable reliably across imports without reloading
    # But we can test the logic path if we force it?
    # Actually, the logic is hardcoded to check _COSY_BACKEND_AVAILABLE.
    # Let's verify explicit python request works.

    MTF._INITIALIZED = False
    MTF.initialize_mtf(max_order=2, max_dimension=2, implementation="python")
    print(f"Explicit Python request chosen: {MTF._IMPLEMENTATION}")
    assert MTF._IMPLEMENTATION == "python", "Explicit python request failed"

    print("Explicit fallback test passed.")


if __name__ == "__main__":
    test_default_backend()
    test_fallback_simulation()
