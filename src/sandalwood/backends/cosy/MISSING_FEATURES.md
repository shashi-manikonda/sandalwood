# Missing Features in COSY Backend

The current implementation of the COSY backend in `cosy_backend.py` relies on `wrapper.f` (Fortran wrapper) and the COSY library. However, there are significant discrepancies between the Python backend expectations and the available Fortran wrapper.

## 1. Missing Library
The `libcosy.so` library is missing, and the source code (`cosy_src`) is a broken symlink. This prevents the backend from running.

## 2. Missing Wrapper Functions
The following functions are used in `cosy_backend.py` but are not defined in `wrapper.f`:

### Memory Management
- `cosy_free_(idx)`: Used to free DA variables. `wrapper.f` does not expose a free function for individual DA variables.

### Initialization
- `cosy_set_coeffs_(idx, coeffs, exps, n_terms)`: Used to initialize a DA variable from a list of coefficients (needed for `from_numpy`). `wrapper.f` has `SET_DA_COEFF` but it is empty/stubbed.

### Complex Number Support
`wrapper.f` does not expose Complex DA (CDA) functions, but `cosy_backend.py` expects them:
- `create_cda_var_`
- `create_cda_const_`
- `compute_cd_add_`
- `compute_cd_sub_`
- `compute_cd_mul_`
- `compute_cd_div_`
- `compute_cd_exp_`
- `compute_cd_sin_`
- `compute_cd_cos_`
- `get_cda_re_`
- `get_cda_im_`
- `set_cd_parts_`
- `compute_da_to_cd_`
- `get_cda_coeff_by_index_`

### Advanced Math Functions
- `compute_da_minv_`
- `compute_da_isrt3_`
- `compute_da_poi_` (Poisson bracket)
- `compute_da_daest_` (Estimation)

## 3. Discrepancies
- `get_da_coeff_by_index_` was renamed to `get_da_term_decoded_` in `cosy_backend.py` to match `wrapper.f`.

## Recommendations
1. Obtain the COSY source code or a compiled `libcosy.so`.
2. Update `wrapper.f` to implement the missing functions (especially `cosy_set_coeffs_` and memory management).
3. If Complex DA is needed, `wrapper.f` must be extended to use COSY's complex support (Common Block `/FOXTYID/ ... NCD`).
