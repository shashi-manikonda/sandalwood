"""
briefdemo_sandalwood.py

A Python translation of the COSY Infinity briefdemo.fox script,
demonstrating core data types and Differential Algebra (DA) operations
using the Sandalwood library.
"""

import os
import sys

import numpy as np
from sandalwood import cmtf, mtf


def pause():
    if not sys.stdin.isatty() or os.environ.get("SANDALWOOD_SKIP_PAUSE") == "1":
        return
    try:
        input("\nEnter any number to continue...")
    except (EOFError, KeyboardInterrupt):
        pass


def cosy_st():
    print("\n** COSY STrings **\n")
    # In COSY: X1 := 'Hello World!'
    x1 = "Hello World!"

    # In COSY: X2 := X1|(7&8) ; X2 := X2&(X1|7)&(X1|LENGTH(X1))
    # COSY indices are 1-based. 7&8 -> 'Wo'
    # 'Wo' & ' ' & '!' -> 'WoW!'
    x2 = x1[6:8]  # 'Wo'
    x2 = x2 + "W" + x1[-1]  # Simple Python concatenation to mirror 'WoW!'

    print(f"{x2} {x1}")
    print("\nThe above string was crafted by manipulating 'Hello World!'.")
    pause()


def cosy_lo():
    print("\n** COSY LOgicals **\n")
    x1 = True
    x2 = False

    # In COSY, logicals are printed as (TRUE ) or (FALSE)
    def st_lo(b):
        return "TRUE " if b else "FALSE"

    print(f"({st_lo(x1)})*({st_lo(x2)}) is {st_lo(x1 and x2)}")
    print(f"({st_lo(x1)})+({st_lo(x2)}) is {st_lo(x1 or x2)}")
    pause()


def cosy_cm(a, b):
    print("\n** COSY CoMplex numbers **\n")
    # In COSY: IM := CM(0&1)
    im = complex(0, 1)
    # In COSY: X1 := CM(A&B)
    x1 = complex(a, b)

    print(f"IM := CM(0&1) is ({im.real:.8f}, {im.imag:.8f})  : imaginary unit")
    print(f"X1 := CM(A&B) is ({x1.real:.8f}, {x1.imag:.8f})")
    print("")
    print(f"Extracting the real      part of X1: {x1.real:.15f}")
    print(f"Extracting the imaginary part of X1: {x1.imag:.15f}")
    pause()


def cosy_ve(a, b):
    print("\n** COSY VEctors **\n")
    # In COSY: X1 := A&B
    x1 = np.array([a, b], dtype=float)
    # In COSY: X2 := X1&X1
    x2 = np.concatenate([x1, x1])

    print(f"X1 := A&B is\n   {x1[0]:.6f}       {x1[1]:.6f}")
    print(
        f"X2 := X1&X1 is\n   {x2[0]:.6f}       {x2[1]:.6f}       {x2[2]:.6f}       {x2[3]:.6f}"
    )
    print("")
    # COSY extracts 1st element as |1, but briefdemo uses X2|3
    print(f"Extracting the component  3      from X2:\n  {x2[2]:.15f}")
    # X2|(2&4) -> elements 2, 3, 4
    slice_vec = x2[1:4]
    print(
        f"Extracting the components 2 to 4 from X2:\n   {slice_vec[0]:.6f}       {slice_vec[1]:.6f}       {slice_vec[2]:.6f}"
    )
    pause()


def cosy_da(a, b):
    print("\n** COSY DA's **\n")

    # Initialize MTF globals
    # In briefdemo.fox: NO := 3 ; NV := 2
    mtf.initialize_mtf(max_order=3, max_dimension=2)

    # In COSY: DA(1)
    da1 = mtf.var(1)
    print("DA(1), the 1-st identity, is")
    print(da1.get_tabular_dataframe().to_string(index=False))

    # In COSY: X1 := A+DA(1) ; X2 := B+DA(2) ; X3 := X1-2*SQR(X2)
    x1 = a + da1
    x2 = b + mtf.var(2)
    x3 = x1 - 2 * (x2**2)

    print("X3 := X1-2*SQR(X2) where X1=A+DA(1), X2=B+DA(2)")
    print(x3.get_tabular_dataframe().to_string(index=False))
    print("")

    # In COSY: X3|(0&1)
    coeff = x3.extract_coefficient((0, 1))
    print(f"Extracting the (0,1) coefficient from X3: {coeff.item():.15f}")
    print("")

    # In COSY: X1 := X3%(-1)  (integral wrt 1st var)
    integral_x3 = x3.integrate(1)
    print("Integral   of X3 w.r.t. the 1st variable")
    print(integral_x3.get_tabular_dataframe().to_string(index=False))

    # In COSY: X2 := X1%1 (derivative wrt 1st var)
    deriv_x3 = integral_x3.deriv(1)
    print("Derivative of it w.r.t. the 1st variable")
    print(deriv_x3.get_tabular_dataframe().to_string(index=False))
    pause()


def cosy_cd():
    print("\n** COSY Complex DA's **\n")
    # In COSY: CD(1)
    cd1 = cmtf.from_variable(1, dimension=2)
    print("CD(1), the 1-st identity, is")
    print(cd1.get_tabular_dataframe().to_string(index=False))
    pause()


def main():
    a, b = 2.0, 3.0
    print(f"\nA={a:.15f}  B={b:.15f}")

    cosy_st()
    cosy_lo()
    cosy_cm(a, b)
    cosy_ve(a, b)
    cosy_da(a, b)
    cosy_cd()


if __name__ == "__main__":
    main()
