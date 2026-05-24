"""
Thread-safety regression tests for sandalwood backend-hardening.

Exercises:
- ``MultivariateTaylorFunction.initialize_mtf`` with concurrent callers
  (guarded by ``_INIT_LOCK`` threading.RLock since feat/backend-hardening).
- ``CosyIndexPool.acquire / release`` with concurrent callers
  (guarded by a threading.RLock since feat/backend-hardening).

These tests use real ``threading.Thread`` instances to provoke the race
conditions that existed before the hardening.  They are designed to be
deterministic: failure before the fix was near-certain, passing after the
fix is certain (the RLock makes the operations atomic).
"""

import threading

import pytest

import sandalwood.taylor_function as tf
from sandalwood.taylor_function import MultivariateTaylorFunction as mtf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COSY_AVAILABLE = tf._COSY_BACKEND_AVAILABLE


def _reset_mtf():
    mtf._INITIALIZED = False
    mtf._MAX_ORDER = None
    mtf._MAX_DIMENSION = None


# ===========================================================================
# initialize_mtf thread-safety
# ===========================================================================


class TestInitializeMtfConcurrent:
    """Guards the _INIT_LOCK added in feat/backend-hardening."""

    def setup_method(self):
        _reset_mtf()

    def teardown_method(self):
        _reset_mtf()

    def test_concurrent_same_params_no_exception(self):
        """
        20 threads each calling initialize_mtf with identical parameters must
        all succeed without raising an exception and leave the library in a
        consistent state.

        Before the RLock, concurrent writers could produce a partially-
        initialised class (e.g. _MAX_ORDER set but _INITIALIZED still False),
        causing later callers to re-enter and corrupt class-level state.
        """
        N_THREADS = 20
        errors = []

        def worker():
            try:
                mtf.initialize_mtf(
                    max_order=4,
                    max_dimension=2,
                    implementation="python",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions raised in worker threads: {errors}"
        # State must be consistent after all threads complete
        assert mtf._INITIALIZED is True
        assert mtf._MAX_ORDER == 4
        assert mtf._MAX_DIMENSION == 2

    def test_concurrent_calls_do_not_corrupt_max_order(self):
        """
        Even when threads call initialize_mtf simultaneously, _MAX_ORDER must
        be set to exactly the value passed in (not a half-written value or 0).
        """
        ORDER = 6
        N_THREADS = 15
        errors = []

        def worker():
            try:
                mtf.initialize_mtf(
                    max_order=ORDER,
                    max_dimension=3,
                    implementation="python",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions in threads: {errors}"
        # The first successful write wins; subsequent calls should be no-ops.
        # Either way, the stored order must equal ORDER (never partial / 0).
        assert mtf._MAX_ORDER == ORDER

    def test_arithmetic_correct_after_concurrent_init(self):
        """
        After concurrent initialisation, basic arithmetic must still yield the
        correct result — confirming that class-level tables are not corrupted.
        """
        N_THREADS = 10
        errors = []

        def worker():
            try:
                mtf.initialize_mtf(
                    max_order=3,
                    max_dimension=2,
                    implementation="python",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

        # Sanity check: (1+x)^2 = 1 + 2x + x^2
        x = mtf.var(1)
        result = (1 + x) ** 2
        import numpy as np

        assert np.isclose(result.extract_coefficient((0, 0)), 1.0)
        assert np.isclose(result.extract_coefficient((1, 0)), 2.0)
        assert np.isclose(result.extract_coefficient((2, 0)), 1.0)


# ===========================================================================
# CosyIndexPool thread-safety
# ===========================================================================


@pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY backend not available")
class TestCosyIndexPoolConcurrent:
    """Guards the threading.RLock added to CosyIndexPool in feat/backend-hardening."""

    def setup_method(self):
        _reset_mtf()
        mtf.initialize_mtf(max_order=3, max_dimension=2, implementation="cosy")

    def teardown_method(self):
        _reset_mtf()

    def test_concurrent_acquire_no_duplicate_indices(self):
        """
        10 threads each acquiring and immediately releasing 50 indices must
        never hand the same index to two threads simultaneously.

        Strategy: each thread records the index it received *before* releasing
        it.  We use a barrier so all acquire calls are as simultaneous as
        possible.  A global set plus a lock detects duplicates.

        Before the RLock, concurrent acquire() calls could pop and re-assign
        the same index position to multiple threads.
        """
        from sandalwood.backends.cosy.cosy_backend import CosyIndexPool

        N_THREADS = 10
        ACQUIRES_PER_THREAD = 50
        barrier = threading.Barrier(N_THREADS)

        in_flight: set[int] = set()
        in_flight_lock = threading.Lock()
        duplicates: list[int] = []
        errors: list[Exception] = []

        def worker():
            try:
                barrier.wait()  # synchronise start
                for _ in range(ACQUIRES_PER_THREAD):
                    idx = CosyIndexPool.acquire()
                    # Check for duplicate while holding our local lock
                    with in_flight_lock:
                        if idx in in_flight:
                            duplicates.append(idx)
                        in_flight.add(idx)
                    # Simulate a tiny computation
                    import time
                    time.sleep(0)
                    with in_flight_lock:
                        in_flight.discard(idx)
                    CosyIndexPool.release(idx)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions in CosyIndexPool threads: {errors}"
        assert duplicates == [], (
            f"CosyIndexPool handed the same index to multiple threads: {duplicates[:5]}"
        )
