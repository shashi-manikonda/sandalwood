---
description: Full QA and Release Verification for Sandalwood
---

Standard procedure to verify releases, ensuring hybrid backend stability.

1.  **Unit Tests**:
    ```bash
    pytest -v
    ```
2.  **Performance Check**:
    Run the multiplication benchmark to detect regressions:
    ```bash
    python scripts/benchmarks/benchmark_multiplication.py
    ```
3.  // turbo
    **Demo Verification**:
    Run demos in Quick Mode:
    ```bash
    pytest -v -m demo tests/test_demos_quick.py
    ```
4.  **Linting & Types**:
    ```bash
    ruff check . && mypy src
    ```
5.  **Documentation**:
    Ensure `API_REFERENCE.md` covers any new operators.
6.  **Version Bump**:
    Update `pyproject.toml` and `CHANGELOG.md`.
