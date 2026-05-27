# Sandalwood ArXiv 2026 Paper

This directory contains the source code and assets for the 2026 ArXiv paper.

## Directory Structure

* `figures/`: Contains all plots, diagrams, and images used in the paper.
* `sections/`: Contains individual `.tex` files for different sections of the paper.
* `scripts/`: Contains Python scripts or Jupyter notebooks used to generate the figures for this paper.

## Building the Paper

To compile the paper into a PDF, run the following commands:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Alternatively, you can use `latexmk`:

```bash
latexmk -pdf main.tex
```

## Reproducing Results

(Add instructions here on how to reproduce the results and figures using the scripts in the `scripts/` directory.)
