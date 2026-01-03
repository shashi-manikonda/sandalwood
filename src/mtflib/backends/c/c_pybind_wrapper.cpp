#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "c_backend.h"

namespace py = pybind11;

py::dict biot_savart_c_from_flat_numpy(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> all_exponents,
    py::array_t<complex_t, py::array::c_style | py::array::forcecast> all_coeffs,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> shapes
) {
    auto exps_buf = all_exponents.request();
    auto coeffs_buf = all_coeffs.request();
    auto shapes_buf = shapes.request();

    const int32_t* exps_ptr = static_cast<const int32_t*>(exps_buf.ptr);
    const complex_t* coeffs_ptr = static_cast<const complex_t*>(coeffs_buf.ptr);
    const int32_t* shapes_ptr = static_cast<const int32_t*>(shapes_buf.ptr);

    int32_t* result_exps_ptr = nullptr;
    complex_t* result_coeffs_ptr = nullptr;
    int32_t* result_shapes_ptr = nullptr;
    size_t total_result_terms = 0;

    biot_savart_c(exps_ptr, coeffs_ptr, shapes_ptr, &result_exps_ptr, &result_coeffs_ptr, &result_shapes_ptr, &total_result_terms);

    int dimension = shapes_ptr[3];
    int n_field_points = shapes_ptr[2];

    py::capsule free_exps(result_exps_ptr, [](void *f) { delete[] static_cast<int32_t*>(f); });
    py::array_t<int32_t> result_exps(
        {static_cast<pybind11::ssize_t>(total_result_terms), static_cast<pybind11::ssize_t>(dimension)},
        {sizeof(int32_t) * dimension, sizeof(int32_t)},
        result_exps_ptr,
        free_exps);

    py::capsule free_coeffs(result_coeffs_ptr, [](void *f) { delete[] static_cast<complex_t*>(f); });
    py::array_t<complex_t> result_coeffs(
        {static_cast<pybind11::ssize_t>(total_result_terms)},
        {sizeof(complex_t)},
        result_coeffs_ptr,
        free_coeffs);

    py::capsule free_shapes(result_shapes_ptr, [](void *f) { delete[] static_cast<int32_t*>(f); });
    py::array_t<int32_t> result_shapes(
        {static_cast<pybind11::ssize_t>(n_field_points * 3 + 1)},
        {sizeof(int32_t)},
        result_shapes_ptr,
        free_shapes);

    py::dict result;
    result["exponents"] = result_exps;
    result["coeffs"] = result_coeffs;
    result["shapes"] = result_shapes;
    return result;
}


PYBIND11_MODULE(mtf_c_backend, m) {
    m.doc() = "C++ backend (C-style interface) for MTFLibrary operations";
    m.def("biot_savart_c_from_flat_numpy", &biot_savart_c_from_flat_numpy, "Biot-Savart law with C-style backend");
}
