#include "c_backend.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>

// --- Helper Functions for MtfDataC ---

MtfDataC create_mtf_data(int dimension, size_t n_terms) {
    MtfDataC mtf;
    mtf.dimension = dimension;
    mtf.n_terms = n_terms;
    if (n_terms > 0) {
        mtf.exponents = new int32_t[n_terms * dimension];
        mtf.coeffs = new complex_t[n_terms];
    } else {
        mtf.exponents = nullptr;
        mtf.coeffs = nullptr;
    }
    return mtf;
}

void free_mtf_data(MtfDataC* mtf) {
    if (mtf) {
        delete[] mtf->exponents;
        delete[] mtf->coeffs;
    }
}

MtfDataC clone_mtf_data(const MtfDataC* src) {
    MtfDataC dst = create_mtf_data(src->dimension, src->n_terms);
    if (src->n_terms > 0) {
        memcpy(dst.exponents, src->exponents, src->n_terms * src->dimension * sizeof(int32_t));
        memcpy(dst.coeffs, src->coeffs, src->n_terms * sizeof(complex_t));
    }
    return dst;
}

// --- Math Operations ---

MtfDataC mtf_negate(const MtfDataC* a) {
    MtfDataC result = clone_mtf_data(a);
    for (size_t i = 0; i < result.n_terms; ++i) {
        result.coeffs[i] = -result.coeffs[i];
    }
    return result;
}

int exponent_compare(const int32_t* exp1, const int32_t* exp2, int dimension) {
    return memcmp(exp1, exp2, dimension * sizeof(int32_t));
}

MtfDataC mtf_add(const MtfDataC* a, const MtfDataC* b) {
    if (!a || !b) return create_mtf_data(0, 0);
    if (a->n_terms == 0) return clone_mtf_data(b);
    if (b->n_terms == 0) return clone_mtf_data(a);

    std::vector<int32_t> new_exponents;
    new_exponents.reserve((a->n_terms + b->n_terms) * a->dimension);
    std::vector<complex_t> new_coeffs;
    new_coeffs.reserve(a->n_terms + b->n_terms);

    size_t i = 0, j = 0;
    while (i < a->n_terms && j < b->n_terms) {
        const int32_t* exp_a = a->exponents + i * a->dimension;
        const int32_t* exp_b = b->exponents + j * b->dimension;
        int cmp = exponent_compare(exp_a, exp_b, a->dimension);

        if (cmp < 0) {
            new_exponents.insert(new_exponents.end(), exp_a, exp_a + a->dimension);
            new_coeffs.push_back(a->coeffs[i]);
            i++;
        } else if (cmp > 0) {
            new_exponents.insert(new_exponents.end(), exp_b, exp_b + a->dimension);
            new_coeffs.push_back(b->coeffs[j]);
            j++;
        } else {
            complex_t sum = a->coeffs[i] + b->coeffs[j];
            if (std::abs(sum) > 1e-16) {
                new_exponents.insert(new_exponents.end(), exp_a, exp_a + a->dimension);
                new_coeffs.push_back(sum);
            }
            i++;
            j++;
        }
    }

    while (i < a->n_terms) {
        const int32_t* exp_a = a->exponents + i * a->dimension;
        new_exponents.insert(new_exponents.end(), exp_a, exp_a + a->dimension);
        new_coeffs.push_back(a->coeffs[i]);
        i++;
    }
    while (j < b->n_terms) {
        const int32_t* exp_b = b->exponents + j * b->dimension;
        new_exponents.insert(new_exponents.end(), exp_b, exp_b + a->dimension);
        new_coeffs.push_back(b->coeffs[j]);
        j++;
    }

    MtfDataC result = create_mtf_data(a->dimension, new_coeffs.size());
    if (!new_coeffs.empty()) {
        memcpy(result.exponents, new_exponents.data(), new_exponents.size() * sizeof(int32_t));
        memcpy(result.coeffs, new_coeffs.data(), new_coeffs.size() * sizeof(complex_t));
    }
    return result;
}

struct Term {
    const int32_t* exponents;
    complex_t coeff;
};

MtfDataC mtf_multiply(const MtfDataC* a, const MtfDataC* b) {
    if (!a || !b || a->n_terms == 0 || b->n_terms == 0) {
        return create_mtf_data(a ? a->dimension : (b ? b->dimension : 0), 0);
    }

    std::vector<int32_t> new_exponents_data(a->n_terms * b->n_terms * a->dimension);
    std::vector<Term> new_terms;
    new_terms.reserve(a->n_terms * b->n_terms);

    for (size_t i = 0; i < a->n_terms; ++i) {
        for (size_t j = 0; j < b->n_terms; ++j) {
            size_t term_idx = i * b->n_terms + j;
            int32_t* current_exp = &new_exponents_data[term_idx * a->dimension];
            for (int k = 0; k < a->dimension; ++k) {
                current_exp[k] = a->exponents[i * a->dimension + k] + b->exponents[j * b->dimension + k];
            }
            new_terms.push_back({current_exp, a->coeffs[i] * b->coeffs[j]});
        }
    }

    std::sort(new_terms.begin(), new_terms.end(), [&](const Term& t1, const Term& t2) {
        return exponent_compare(t1.exponents, t2.exponents, a->dimension) < 0;
    });

    std::vector<int32_t> final_exponents;
    std::vector<complex_t> final_coeffs;
    if (!new_terms.empty()) {
        final_exponents.insert(final_exponents.end(), new_terms[0].exponents, new_terms[0].exponents + a->dimension);
        final_coeffs.push_back(new_terms[0].coeff);

        for (size_t i = 1; i < new_terms.size(); ++i) {
            if (exponent_compare(new_terms[i].exponents, &final_exponents.back() - a->dimension + 1, a->dimension) == 0) {
                final_coeffs.back() += new_terms[i].coeff;
            } else {
                final_exponents.insert(final_exponents.end(), new_terms[i].exponents, new_terms[i].exponents + a->dimension);
                final_coeffs.push_back(new_terms[i].coeff);
            }
        }
    }

    MtfDataC result = create_mtf_data(a->dimension, final_coeffs.size());
    if(!final_coeffs.empty()) {
        memcpy(result.exponents, final_exponents.data(), final_exponents.size() * sizeof(int32_t));
        memcpy(result.coeffs, final_coeffs.data(), final_coeffs.size() * sizeof(complex_t));
    }
    return result;
}

static const double isqrt_coeffs[] = {
    1.0, -0.5, 0.375, -0.3125, 0.2734375, -0.24609375, 0.2255859375, -0.20947265625,
    0.196380615234375, -0.1854705810546875, 0.17619705200195312, -0.16818809509277344,
    0.1611802577972412, -0.15498101711273193, 0.14944598078727722, -0.14446444809436798
};
static const int num_isqrt_coeffs = sizeof(isqrt_coeffs) / sizeof(double);

MtfDataC mtf_power(const MtfDataC* a, double exponent) {
    if (exponent != -0.5) {
        int exp_int = static_cast<int>(exponent);
        if (exp_int == exponent && exp_int >= 0) { // integer power
            if (exp_int == 0) {
                MtfDataC result = create_mtf_data(a->dimension, 1);
                memset(result.exponents, 0, result.dimension * sizeof(int32_t));
                result.coeffs[0] = 1.0;
                return result;
            }
            MtfDataC result = clone_mtf_data(a);
            for (int i = 1; i < exp_int; ++i) {
                MtfDataC temp = mtf_multiply(&result, a);
                free_mtf_data(&result);
                result = temp;
            }
            return result;
        }
        return create_mtf_data(a->dimension, 0); // Not implemented for other exponents
    }

    complex_t const_term = 0.0;
    int const_term_idx = -1;
    std::vector<int32_t> zero_exp(a->dimension, 0);
    for (size_t i = 0; i < a->n_terms; ++i) {
        if (memcmp(a->exponents + i * a->dimension, zero_exp.data(), a->dimension * sizeof(int32_t)) == 0) {
            const_term = a->coeffs[i];
            const_term_idx = i;
            break;
        }
    }

    if (const_term_idx == -1 || std::abs(const_term) < 1e-16) {
        return create_mtf_data(a->dimension, 0);
    }

    MtfDataC temp = clone_mtf_data(a);
    complex_t const_factor_isqrt = 1.0 / std::sqrt(const_term);
    for (size_t i = 0; i < temp.n_terms; ++i) {
        temp.coeffs[i] /= const_term;
    }

    MtfDataC x = create_mtf_data(temp.dimension, temp.n_terms > 0 ? temp.n_terms - 1 : 0);
    size_t current_x_term = 0;
    for (size_t i = 0; i < temp.n_terms; ++i) {
        if (i == static_cast<size_t>(const_term_idx)) continue;
        memcpy(x.exponents + current_x_term * x.dimension, temp.exponents + i * temp.dimension, x.dimension * sizeof(int32_t));
        x.coeffs[current_x_term] = temp.coeffs[i];
        current_x_term++;
    }
    free_mtf_data(&temp);

    MtfDataC result = create_mtf_data(a->dimension, 1);
    memset(result.exponents, 0, result.dimension * sizeof(int32_t));
    result.coeffs[0] = isqrt_coeffs[0];

    MtfDataC x_power_n = clone_mtf_data(&x);

    for (int n = 1; n < num_isqrt_coeffs; ++n) {
        MtfDataC term = clone_mtf_data(&x_power_n);
        for (size_t i = 0; i < term.n_terms; ++i) {
            term.coeffs[i] *= isqrt_coeffs[n];
        }
        MtfDataC new_result = mtf_add(&result, &term);
        free_mtf_data(&result);
        result = new_result;
        free_mtf_data(&term);

        if (n < num_isqrt_coeffs - 1) {
            MtfDataC new_x_power_n = mtf_multiply(&x_power_n, &x);
            free_mtf_data(&x_power_n);
            x_power_n = new_x_power_n;
        }
    }

    free_mtf_data(&x);
    free_mtf_data(&x_power_n);

    for (size_t i = 0; i < result.n_terms; ++i) {
        result.coeffs[i] *= const_factor_isqrt;
    }

    return result;
}

// --- Main Biot-Savart Function ---

void biot_savart_c(
    const int32_t* all_exponents, const complex_t* all_coeffs, const int32_t* shapes,
    int32_t** result_exps, complex_t** result_coeffs, int32_t** result_shapes, size_t* total_result_terms_out
) {
    int n_source_points = shapes[0];
    int n_dl_vectors = shapes[1];
    int n_field_points = shapes[2];
    int dimension = shapes[3];
    const int32_t* sp_shapes = shapes + 4;
    const int32_t* dl_shapes = shapes + 4 + n_source_points * 3;
    const int32_t* fp_shapes = shapes + 4 + n_source_points * 3 + n_dl_vectors * 3;

    std::vector<MtfDataC> sp(n_source_points * 3);
    std::vector<MtfDataC> dl(n_dl_vectors * 3);
    std::vector<MtfDataC> fp(n_field_points * 3);

    size_t current_term_offset = 0;
    for(size_t i=0; i < (size_t)n_source_points*3; ++i) {
        sp[i] = create_mtf_data(dimension, sp_shapes[i]);
        if (sp_shapes[i] > 0) {
            memcpy(sp[i].exponents, all_exponents + current_term_offset * dimension, (size_t)sp_shapes[i] * dimension * sizeof(int32_t));
            memcpy(sp[i].coeffs, all_coeffs + current_term_offset, (size_t)sp_shapes[i] * sizeof(complex_t));
            current_term_offset += sp_shapes[i];
        }
    }
    for(size_t i=0; i < (size_t)n_dl_vectors*3; ++i) {
        dl[i] = create_mtf_data(dimension, dl_shapes[i]);
        if (dl_shapes[i] > 0) {
            memcpy(dl[i].exponents, all_exponents + current_term_offset * dimension, (size_t)dl_shapes[i] * dimension * sizeof(int32_t));
            memcpy(dl[i].coeffs, all_coeffs + current_term_offset, (size_t)dl_shapes[i] * sizeof(complex_t));
            current_term_offset += dl_shapes[i];
        }
    }
    for(size_t i=0; i < (size_t)n_field_points*3; ++i) {
        fp[i] = create_mtf_data(dimension, fp_shapes[i]);
        if (fp_shapes[i] > 0) {
            memcpy(fp[i].exponents, all_exponents + current_term_offset * dimension, (size_t)fp_shapes[i] * dimension * sizeof(int32_t));
            memcpy(fp[i].coeffs, all_coeffs + current_term_offset, (size_t)fp_shapes[i] * sizeof(complex_t));
            current_term_offset += fp_shapes[i];
        }
    }

    std::vector<MtfDataC> B_fields(n_field_points * 3);
    double mu_0_4pi = 1e-7;

    #pragma omp parallel for
    for (int i = 0; i < n_field_points; ++i) {
        MtfDataC B_field_total[3];
        for(int k=0; k<3; ++k) B_field_total[k] = create_mtf_data(dimension, 0);

        for (int j = 0; j < n_source_points; ++j) {
            MtfDataC r_vec[3];
            for(int k=0; k<3; ++k) {
                MtfDataC neg_sp = mtf_negate(&sp[j*3+k]);
                r_vec[k] = mtf_add(&fp[i*3+k], &neg_sp);
                free_mtf_data(&neg_sp);
            }

            MtfDataC r2 = mtf_multiply(&r_vec[0], &r_vec[0]);
            MtfDataC r2y = mtf_multiply(&r_vec[1], &r_vec[1]);
            MtfDataC r2z = mtf_multiply(&r_vec[2], &r_vec[2]);
            MtfDataC r2_sum1 = mtf_add(&r2, &r2y);
            MtfDataC r2_sum2 = mtf_add(&r2_sum1, &r2z);

            MtfDataC r_inv = mtf_power(&r2_sum2, -0.5);
            MtfDataC r_inv2 = mtf_multiply(&r_inv, &r_inv);
            MtfDataC inv_r3 = mtf_multiply(&r_inv2, &r_inv);

            MtfDataC cross_prod[3];
            MtfDataC dl_x_r_x = mtf_multiply(&dl[j*3+1], &r_vec[2]);
            MtfDataC dl_y_r_x = mtf_multiply(&dl[j*3+2], &r_vec[1]);
            MtfDataC neg_dl_y_r_x = mtf_negate(&dl_y_r_x);
            cross_prod[0] = mtf_add(&dl_x_r_x, &neg_dl_y_r_x);

            MtfDataC dl_z_r_x = mtf_multiply(&dl[j*3+2], &r_vec[0]);
            MtfDataC dl_x_r_z = mtf_multiply(&dl[j*3+0], &r_vec[2]);
            MtfDataC neg_dl_x_r_z = mtf_negate(&dl_x_r_z);
            cross_prod[1] = mtf_add(&dl_z_r_x, &neg_dl_x_r_z);

            MtfDataC dl_x_r_y = mtf_multiply(&dl[j*3+0], &r_vec[1]);
            MtfDataC dl_y_r_x_2 = mtf_multiply(&dl[j*3+1], &r_vec[0]);
            MtfDataC neg_dl_y_r_x_2 = mtf_negate(&dl_y_r_x_2);
            cross_prod[2] = mtf_add(&dl_x_r_y, &neg_dl_y_r_x_2);

            for(int k=0; k<3; ++k) {
                MtfDataC term = mtf_multiply(&cross_prod[k], &inv_r3);
                for(size_t l=0; l<term.n_terms; ++l) term.coeffs[l] *= mu_0_4pi;
                MtfDataC new_total = mtf_add(&B_field_total[k], &term);
                free_mtf_data(&B_field_total[k]);
                B_field_total[k] = new_total;
                free_mtf_data(&term);
            }
        }
        for(int k=0; k<3; ++k) B_fields[i*3+k] = B_field_total[k];
    }

    size_t total_terms = 0;
    for(int i=0; i<n_field_points*3; ++i) total_terms += B_fields[i].n_terms;
    *total_result_terms_out = total_terms;

    *result_exps = new int32_t[total_terms * dimension];
    *result_coeffs = new complex_t[total_terms];
    *result_shapes = new int32_t[n_field_points * 3 + 1];
    (*result_shapes)[0] = n_field_points;

    size_t current_res_offset = 0;
    for(int i=0; i<n_field_points*3; ++i) {
        (*result_shapes)[i+1] = B_fields[i].n_terms;
        if (B_fields[i].n_terms > 0) {
            memcpy(*result_exps + current_res_offset * dimension, B_fields[i].exponents, B_fields[i].n_terms * dimension * sizeof(int32_t));
            memcpy(*result_coeffs + current_res_offset, B_fields[i].coeffs, B_fields[i].n_terms * sizeof(complex_t));
            current_res_offset += B_fields[i].n_terms;
        }
    }

    for(auto& mtf : sp) free_mtf_data(&mtf);
    for(auto& mtf : dl) free_mtf_data(&mtf);
    for(auto& mtf : fp) free_mtf_data(&mtf);
    for(auto& mtf : B_fields) free_mtf_data(&mtf);
}
