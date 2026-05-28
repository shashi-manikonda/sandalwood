#!/usr/bin/env bash
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Sandalwood LaTeX Compilation Script ==="

# Temporary downloads list to clean up after building
STYS=(
    "revtex4-2.cls"
    "revsymb4-2.sty"
    "ltxdocext.sty"
    "ltxutil.sty"
    "ltxfront.sty"
    "ltxgrid.sty"
    "aps10pt4-2.rtx"
    "aps11pt4-2.rtx"
    "aps12pt4-2.rtx"
    "aps4-2.rtx"
    "apsrmp4-2.rtx"
    "aip4-2.rtx"
    "aapm4-2.rtx"
    "sor4-2.rtx"
    "textcase.sty"
    "listings.sty"
    "xcolor.sty"
    "etoolbox.sty"
    "etoolbox.def"
    "booktabs.sty"
)

# Function to download and extract CTAN packages
prepare_dependencies() {
    echo "Checking LaTeX package dependencies..."
    
    # 1. REVTeX 4.2
    if [ ! -f "revtex4-2.cls" ]; then
        echo "Downloading REVTeX 4.2 from CTAN..."
        curl -L -o revtex.zip https://mirrors.ctan.org/macros/latex/contrib/revtex.zip
        python3 -c "import zipfile; zipfile.ZipFile('revtex.zip').extractall('revtex_extracted')"
        (
            cd revtex_extracted/revtex
            tex revtex4-2.dtx
            tex aip4-2.dtx
            tex ltxdocext.dtx
            tex ltxutil.dtx
            tex ltxfront.dtx
            tex ltxgrid.dtx
        )
        cp revtex_extracted/revtex/*.cls revtex_extracted/revtex/*.sty revtex_extracted/revtex/*.rtx .
        rm -rf revtex.zip revtex_extracted
    fi

    # 2. textcase
    if [ ! -f "textcase.sty" ]; then
        echo "Downloading textcase from CTAN..."
        curl -L -o textcase.zip https://mirrors.ctan.org/macros/latex/contrib/textcase.zip
        python3 -c "import zipfile; zipfile.ZipFile('textcase.zip').extractall('textcase_extracted')"
        (
            cd textcase_extracted/textcase
            tex textcase.ins
        )
        cp textcase_extracted/textcase/textcase.sty .
        rm -rf textcase.zip textcase_extracted
    fi

    # 3. listings
    if [ ! -f "listings.sty" ]; then
        echo "Downloading listings from CTAN..."
        curl -L -o listings.zip https://mirrors.ctan.org/macros/latex/contrib/listings.zip
        python3 -c "import zipfile; zipfile.ZipFile('listings.zip').extractall('listings_extracted')"
        (
            cd listings_extracted/listings
            tex listings.ins
        )
        cp listings_extracted/listings/*.sty listings_extracted/listings/*.cfg listings_extracted/listings/*.prf .
        rm -rf listings.zip listings_extracted
    fi

    # 4. xcolor
    if [ ! -f "xcolor.sty" ]; then
        echo "Downloading xcolor from CTAN..."
        curl -L -o xcolor.zip https://mirrors.ctan.org/macros/latex/contrib/xcolor.zip
        python3 -c "import zipfile; zipfile.ZipFile('xcolor.zip').extractall('xcolor_extracted')"
        (
            cd xcolor_extracted/xcolor
            tex xcolor.ins
        )
        cp xcolor_extracted/xcolor/xcolor.sty .
        rm -rf xcolor.zip xcolor_extracted
    fi

    # 5. etoolbox
    if [ ! -f "etoolbox.sty" ]; then
        echo "Downloading etoolbox from CTAN..."
        curl -L -o etoolbox.zip https://mirrors.ctan.org/macros/latex/contrib/etoolbox.zip
        python3 -c "import zipfile; zipfile.ZipFile('etoolbox.zip').extractall('etoolbox_extracted')"
        cp etoolbox_extracted/etoolbox/etoolbox.sty etoolbox_extracted/etoolbox/etoolbox.def .
        rm -rf etoolbox.zip etoolbox_extracted
    fi

    # 6. booktabs
    if [ ! -f "booktabs.sty" ]; then
        echo "Downloading booktabs from CTAN..."
        curl -L -o booktabs.zip https://mirrors.ctan.org/macros/latex/contrib/booktabs.zip
        python3 -c "import zipfile; zipfile.ZipFile('booktabs.zip').extractall('booktabs_extracted')"
        (
            cd booktabs_extracted/booktabs
            tex booktabs.ins
        )
        cp booktabs_extracted/booktabs/booktabs.sty .
        rm -rf booktabs.zip booktabs_extracted
    fi
}

# Run preparation
prepare_dependencies

# Compile LaTeX to PDF
echo "Compiling main.tex..."
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Clean up auxiliary compilation files
echo "Cleaning up auxiliary files..."
rm -f main.aux main.log main.out main.toc main.synctex.gz

# Clean up style files and other local dependencies to keep the repo clean
echo "Cleaning up downloaded packages..."
for file in "${STYS[@]}"; do
    rm -f "$file"
done
rm -f listings.cfg lstdoc.sty lstlang1.sty lstlang2.sty lstlang3.sty lstmisc.sty lstpatch.sty ltxdoc.cfg
rm -f listings-acm.prf listings-bash.prf listings-fortran.prf listings-hansl.prf listings-lua.prf listings-python.prf listings-rexx.prf listings-riscv.prf

echo "=== Build Complete! main.pdf is updated. ==="
