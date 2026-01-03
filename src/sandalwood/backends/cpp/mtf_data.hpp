#ifndef MTF_DATA_HPP
#define MTF_DATA_HPP

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

class MtfData {
public:
  std::vector<int32_t> exponents; // Flattened 2D array (row-major)
  std::vector<std::complex<double>> coeffs;
  int dimension;
  size_t n_terms;

  // Constructors
  MtfData();
  MtfData(int dim);
  MtfData(const MtfData &other);            // Copy constructor
  MtfData &operator=(const MtfData &other); // Copy assignment

  // Methods
  void reserve(size_t capacity);
  void clear();

  // In-place operations
  void add_inplace(const MtfData &other);
  void subtract_inplace(const MtfData &other);
  void multiply_inplace(const MtfData &other);
  void power_inplace(double exponent); // Initially for -0.5

  // Non-in-place operations
  MtfData add(const MtfData &other) const;
  MtfData subtract(const MtfData &other) const;
  MtfData multiply(const MtfData &other) const;
  MtfData negate() const;

  // Conversion to/from Python
  py::dict to_dict() const;
  void from_numpy(py::array_t<int32_t> exps_arr,
                  py::array_t<std::complex<double>> coeffs_arr);
};

#endif // MTF_DATA_HPP
