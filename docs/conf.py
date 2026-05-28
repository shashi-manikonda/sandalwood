# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))


import re
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sandalwood"
copyright = "2025, Shashikant Manikonda"
author = "Shashikant Manikonda"

# Extract version dynamically from pyproject.toml
pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
version_match = re.search(r'^version\s*=\s*[\'"]([^\'"]+)[\'"]', pyproject_path.read_text(encoding="utf-8"), re.M)

version = version_match.group(1) if version_match else "unknown"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "nbsphinx",
]

source_suffix = [".rst", ".md"]

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = "always"

# -- autosummary configuration -----------------------------------------------
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "demos/README.md"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Enable the dollarmath extension to use $...$ for inline math
myst_enable_extensions = ["dollarmath"]
