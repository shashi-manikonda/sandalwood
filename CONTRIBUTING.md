# Contributing to mtflib

First off, thank you for considering contributing to `mtflib`! Your help is greatly appreciated. This document provides guidelines for contributing to the project.

## How to Report a Bug

If you find a bug, please open an issue on our [GitHub Issue Tracker](https://github.com/sm-physics/mtflib/issues).

When reporting a bug, please include the following:

*   **A clear and descriptive title.**
*   **A detailed description of the bug,** including the steps to reproduce it.
*   **A code snippet** that demonstrates the bug.
*   **The expected behavior** and what you observed instead.
*   **Your system information,** including your Python version and the version of `mtflib` you are using.

## How to Suggest a New Feature

We are always open to new ideas! If you have a suggestion for a new feature or an enhancement to an existing one, please open an issue on our [GitHub Issue Tracker](https://github.com/sm-physics/mtflib/issues).

When suggesting a feature, please include:

*   **A clear and descriptive title.**
*   **A detailed description of the proposed feature** and why it would be valuable to the project.
*   **A code example or pseudo-code** demonstrating how the feature might be used.

## Setting Up the Development Environment

To contribute code to `mtflib`, you will need to set up a local development environment.

1.  **Fork and Clone the Repository:**
    First, fork the repository on GitHub. Then, clone your fork to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/mtflib.git
    cd mtflib
    ```

2.  **Create a Virtual Environment:**
    It is highly recommended to work within a Python virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Install the library and all its development dependencies, including those for testing and building the documentation.
    ```bash
    uv pip install -e .[dev]
    ```
    The `-e` flag installs the package in "editable" mode, so any changes you make to the source code will be immediately effective.

4.  **Run Tests:**
    Before making any changes, run the test suite to ensure everything is working correctly.
    ```bash
    pytest
    ```

5.  **Make Your Changes:**
    Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b your-feature-name
    ```
    Now you can make your changes to the code.

6.  **Submit a Pull Request:**
    Once you are happy with your changes, push your branch to your fork and open a pull request against the `main` branch of the original repository.
