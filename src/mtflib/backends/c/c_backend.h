#ifndef C_BACKEND_H
#define C_BACKEND_H

#include <cstdint>
#include <cstddef>
#include <complex>

typedef std::complex<double> complex_t;

struct MtfDataC {
    int32_t* exponents;
    complex_t* coeffs;
    size_t n_terms;
    int dimension;
};

void biot_savart_c(
    const int32_t* all_exponents, const complex_t* all_coeffs, const int32_t* shapes,
    int32_t** result_exps, complex_t** result_coeffs, int32_t** result_shapes, size_t* total_result_terms
);

#endif // C_BACKEND_H
