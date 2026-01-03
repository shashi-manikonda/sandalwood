#include "mtf_data.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(mtf_cpp, m) {
  m.doc() = "C++ backend for MTFLibrary operations";

  py::class_<MtfData>(m, "MtfData")
      .def(py::init<>())
      .def(py::init<int>())
      .def("to_dict", &MtfData::to_dict,
           "Converts the MtfData object to a dictionary of numpy arrays.")
      .def("from_numpy", &MtfData::from_numpy,
           "Initializes MtfData from numpy arrays.")
      .def("add", &MtfData::add, "Adds another MtfData object.")
      .def("add_inplace", &MtfData::add_inplace,
           "Adds another MtfData object in-place.")
      .def("multiply", &MtfData::multiply,
           "Multiplies by another MtfData object.")
      .def("multiply_inplace", &MtfData::multiply_inplace,
           "Multiplies by another MtfData object in-place.")
      .def("negate", &MtfData::negate, "Negates the MtfData object.");

  m.def(
      "switch_backend",
      [](const std::string &backend_name) {
        py::module_::import("mtflib")
            .attr("MultivariateTaylorFunction")
            .attr("_IMPLEMENTATION") = backend_name;
      },
      "Switch the backend implementation");

  m.def(
      "biot_savart_from_flat_numpy",
      [](py::array_t<int32_t> all_exponents,
         py::array_t<std::complex<double>> all_coeffs,
         py::array_t<int32_t> shapes) {
        auto exps_r = all_exponents.unchecked<2>();
        auto coeffs_r = all_coeffs.unchecked<1>();
        auto shapes_r = shapes.unchecked<1>();

        int num_sources = shapes_r(0);
        int num_dls = shapes_r(1);
        int num_fields = shapes_r(2);
        int dimension = shapes_r(3);

        int current_shape_idx = 4;
        int current_data_idx = 0;

        auto load_vectors = [&](int count) {
          std::vector<std::vector<MtfData>> vectors(count);
          for (int i = 0; i < count; ++i) {
            vectors[i].reserve(3);
            for (int dim = 0; dim < 3; ++dim) {
              int n_terms = shapes_r(current_shape_idx++);
              MtfData mtf(dimension);
              mtf.n_terms = n_terms;
              mtf.exponents.resize(n_terms * dimension);
              mtf.coeffs.resize(n_terms);

              for (int t = 0; t < n_terms; ++t) {
                for (int d = 0; d < dimension; ++d) {
                  mtf.exponents[t * dimension + d] =
                      exps_r(current_data_idx + t, d);
                }
                mtf.coeffs[t] = coeffs_r(current_data_idx + t);
              }
              current_data_idx += n_terms;
              vectors[i].push_back(std::move(mtf));
            }
          }
          return vectors;
        };

        auto source_points = load_vectors(num_sources);
        auto dl_vectors = load_vectors(num_dls);
        auto field_points = load_vectors(num_fields);

        // Computation
        std::vector<std::vector<py::dict>> results;
        results.reserve(num_fields);

        double mu0_4pi = 1e-7;

        // Pre-allocate working buffers
        std::vector<MtfData> r_vec(3, MtfData(dimension));
        MtfData r_sq(dimension);

        MtfData temp_mul(dimension); // For intermediate multiplies
        MtfData cp_term1(dimension);
        MtfData cp_term2(dimension);

        MtfData cp_x(dimension);
        MtfData cp_y(dimension);
        MtfData cp_z(dimension);

        MtfData r_inv(dimension);
        MtfData r_inv_3(dimension);

        // Reserve some initial capacity to reduce reallocations during
        // polynomial growth
        int estimated_terms = 50;
        for (int k = 0; k < 3; ++k)
          r_vec[k].reserve(estimated_terms);
        r_sq.reserve(estimated_terms);
        temp_mul.reserve(estimated_terms);
        cp_term1.reserve(estimated_terms);
        cp_term2.reserve(estimated_terms);
        cp_x.reserve(estimated_terms);
        cp_y.reserve(estimated_terms);
        cp_z.reserve(estimated_terms);
        r_inv.reserve(estimated_terms);
        r_inv_3.reserve(estimated_terms);

        for (int i = 0; i < num_fields; ++i) {
          std::vector<MtfData> B(3, MtfData(dimension));
          // Initialize B with zero
          for (int k = 0; k < 3; ++k) {
            B[k].n_terms = 1;
            B[k].coeffs.assign(1, 0.0);
            B[k].exponents.assign(dimension, 0);
            B[k].reserve(estimated_terms * 10); // Accumulator grows
          }

          for (int j = 0; j < num_sources; ++j) {
            // R = Field[i] - Source[j]

            // Reset r_sq to 0
            r_sq.n_terms = 1;
            r_sq.coeffs.assign(1, 0.0);
            r_sq.exponents.assign(dimension, 0);

            for (int k = 0; k < 3; ++k) {
              // r_vec[k] = field_points[i][k] - source_points[j][k]
              // We use assignment then subtract_inplace
              r_vec[k] = field_points[i][k];
              r_vec[k].subtract_inplace(source_points[j][k]);

              // term = r_vec[k] * r_vec[k]
              temp_mul = r_vec[k];
              temp_mul.multiply_inplace(r_vec[k]);

              r_sq.add_inplace(temp_mul);
            }

            // r_inv_3 = (r^2)^-1.5
            // r_sq.power_inplace works in-place
            // We need a copy of r_sq first because we need it to be r^2 for
            // power Actually power_inplace updates r_sq itself. But wait, r_sq
            // is reused next iteration. r_sq holds |r|^2. r_inv = r_sq^-0.5
            r_inv = r_sq;
            r_inv.power_inplace(-0.5);

            // r_inv_3 = r_inv * r_inv * r_inv
            r_inv_3 = r_inv;
            r_inv_3.multiply_inplace(r_inv);
            r_inv_3.multiply_inplace(r_inv);

            // Cross Product
            // Bx += (dly * rz - dlz * ry) * r_inv_3

            // cp_x = dly * rz
            cp_term1 = dl_vectors[j][1];
            cp_term1.multiply_inplace(r_vec[2]);

            // term2 = dlz * ry
            cp_term2 = dl_vectors[j][2];
            cp_term2.multiply_inplace(r_vec[1]);

            cp_x = cp_term1;
            cp_x.subtract_inplace(cp_term2);
            cp_x.multiply_inplace(r_inv_3);
            B[0].add_inplace(cp_x);

            // By += (dlz * rx - dlx * rz) * r_inv_3
            cp_term1 = dl_vectors[j][2];
            cp_term1.multiply_inplace(r_vec[0]);

            cp_term2 = dl_vectors[j][0];
            cp_term2.multiply_inplace(r_vec[2]);

            cp_y = cp_term1;
            cp_y.subtract_inplace(cp_term2);
            cp_y.multiply_inplace(r_inv_3);
            B[1].add_inplace(cp_y);

            // Bz += (dlx * ry - dly * rx) * r_inv_3
            cp_term1 = dl_vectors[j][0];
            cp_term1.multiply_inplace(r_vec[1]);

            cp_term2 = dl_vectors[j][1];
            cp_term2.multiply_inplace(r_vec[0]);

            cp_z = cp_term1;
            cp_z.subtract_inplace(cp_term2);
            cp_z.multiply_inplace(r_inv_3);
            B[2].add_inplace(cp_z);
          }

          std::vector<py::dict> field_point_result;
          for (int k = 0; k < 3; ++k) {
            // Apply mu0/4pi
            for (auto &c : B[k].coeffs)
              c *= mu0_4pi;
            field_point_result.push_back(B[k].to_dict());
          }
          results.push_back(field_point_result);
        }

        return results;
      },
      "vectorized biot savart calculation");
}
