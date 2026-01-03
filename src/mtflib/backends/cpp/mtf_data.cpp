#include "mtf_data.hpp"
#include "precomputed_coefficients.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

// Constructors
MtfData::MtfData() : dimension(0), n_terms(0) {}

MtfData::MtfData(int dim) : dimension(dim), n_terms(0) {}

MtfData::MtfData(const MtfData &other)
    : exponents(other.exponents), coeffs(other.coeffs),
      dimension(other.dimension), n_terms(other.n_terms) {}

MtfData &MtfData::operator=(const MtfData &other) {
  if (this != &other) {
    exponents = other.exponents;
    coeffs = other.coeffs;
    dimension = other.dimension;
    n_terms = other.n_terms;
  }
  return *this;
}

// Methods
void MtfData::reserve(size_t capacity) {
  exponents.reserve(capacity * dimension);
  coeffs.reserve(capacity);
}

void MtfData::clear() {
  exponents.clear();
  coeffs.clear();
  n_terms = 0;
}

// Conversion to/from Python
py::dict MtfData::to_dict() const {
  py::dict d;
  py::array_t<int32_t> exps_arr({(ssize_t)n_terms, (ssize_t)dimension});
  py::array_t<std::complex<double>> coeffs_arr(n_terms);

  auto exps_ptr = exps_arr.mutable_unchecked<2>();
  auto coeffs_ptr = coeffs_arr.mutable_unchecked<1>();

  for (size_t i = 0; i < n_terms; ++i) {
    for (int j = 0; j < dimension; ++j) {
      exps_ptr(i, j) = exponents[i * dimension + j];
    }
    coeffs_ptr(i) = coeffs[i];
  }

  d["exponents"] = exps_arr;
  d["coeffs"] = coeffs_arr;
  return d;
}

void MtfData::from_numpy(py::array_t<int32_t> exps_arr,
                         py::array_t<std::complex<double>> coeffs_arr) {
  auto exps_buf = exps_arr.request();
  auto coeffs_buf = coeffs_arr.request();

  if (exps_buf.ndim != 2 || coeffs_buf.ndim != 1) {
    throw std::runtime_error(
        "Number of dimensions must be 2 for exponents and 1 for coefficients.");
  }
  if (exps_buf.shape[0] != coeffs_buf.shape[0]) {
    throw std::runtime_error("Input shapes must match.");
  }

  dimension = exps_buf.shape[1];
  n_terms = exps_buf.shape[0];
  auto exps_ptr = static_cast<int32_t *>(exps_buf.ptr);
  auto coeffs_ptr = static_cast<std::complex<double> *>(coeffs_buf.ptr);

  exponents.resize(n_terms * dimension);
  coeffs.resize(n_terms);

  std::copy(exps_ptr, exps_ptr + n_terms * dimension, exponents.begin());
  std::copy(coeffs_ptr, coeffs_ptr + n_terms, coeffs.begin());
}

void MtfData::add_inplace(const MtfData &other) {
  if (dimension != other.dimension) {
    throw std::runtime_error("Dimensions must match for addition.");
  }
  if (other.n_terms == 0) {
    return;
  }
  if (n_terms == 0) {
    *this = other;
    return;
  }

  // Create index vectors to sort based on exponents
  std::vector<size_t> this_indices(n_terms);
  std::iota(this_indices.begin(), this_indices.end(), 0);

  std::vector<size_t> other_indices(other.n_terms);
  std::iota(other_indices.begin(), other_indices.end(), 0);

  // Sort the index vectors
  std::sort(this_indices.begin(), this_indices.end(), [&](size_t a, size_t b) {
    return std::lexicographical_compare(exponents.begin() + a * dimension,
                                        exponents.begin() + (a + 1) * dimension,
                                        exponents.begin() + b * dimension,
                                        exponents.begin() +
                                            (b + 1) * dimension);
  });

  std::sort(other_indices.begin(), other_indices.end(),
            [&](size_t a, size_t b) {
              return std::lexicographical_compare(
                  other.exponents.begin() + a * other.dimension,
                  other.exponents.begin() + (a + 1) * other.dimension,
                  other.exponents.begin() + b * other.dimension,
                  other.exponents.begin() + (b + 1) * other.dimension);
            });

  std::vector<int32_t> new_exponents;
  std::vector<std::complex<double>> new_coeffs;
  new_exponents.reserve(n_terms + other.n_terms);
  new_coeffs.reserve(n_terms + other.n_terms);

  size_t i = 0, j = 0;
  while (i < n_terms && j < other.n_terms) {
    auto this_exp_start = exponents.begin() + this_indices[i] * dimension;
    auto this_exp_end = this_exp_start + dimension;
    auto other_exp_start =
        other.exponents.begin() + other_indices[j] * other.dimension;
    auto other_exp_end = other_exp_start + other.dimension;

    if (std::lexicographical_compare(this_exp_start, this_exp_end,
                                     other_exp_start, other_exp_end)) {
      new_exponents.insert(new_exponents.end(), this_exp_start, this_exp_end);
      new_coeffs.push_back(coeffs[this_indices[i]]);
      i++;
    } else if (std::lexicographical_compare(other_exp_start, other_exp_end,
                                            this_exp_start, this_exp_end)) {
      new_exponents.insert(new_exponents.end(), other_exp_start, other_exp_end);
      new_coeffs.push_back(other.coeffs[other_indices[j]]);
      j++;
    } else {
      std::complex<double> sum =
          coeffs[this_indices[i]] + other.coeffs[other_indices[j]];
      if (std::abs(sum.real()) > 1e-16 || std::abs(sum.imag()) > 1e-16) {
        new_exponents.insert(new_exponents.end(), this_exp_start, this_exp_end);
        new_coeffs.push_back(sum);
      }
      i++;
      j++;
    }
  }

  while (i < n_terms) {
    auto this_exp_start = exponents.begin() + this_indices[i] * dimension;
    new_exponents.insert(new_exponents.end(), this_exp_start,
                         this_exp_start + dimension);
    new_coeffs.push_back(coeffs[this_indices[i]]);
    i++;
  }
  while (j < other.n_terms) {
    auto other_exp_start =
        other.exponents.begin() + other_indices[j] * other.dimension;
    new_exponents.insert(new_exponents.end(), other_exp_start,
                         other_exp_start + other.dimension);
    new_coeffs.push_back(other.coeffs[other_indices[j]]);
    j++;
  }

  coeffs.swap(new_coeffs);
  n_terms = coeffs.size();
}

void MtfData::subtract_inplace(const MtfData &other) {
  if (dimension != other.dimension) {
    throw std::runtime_error("Dimensions must match for subtraction.");
  }
  if (other.n_terms == 0) {
    return;
  }
  if (n_terms == 0) {
    *this = other;
    for (auto &c : coeffs)
      c = -c; // Negate
    return;
  }

  // Create index vectors to sort based on exponents
  std::vector<size_t> this_indices(n_terms);
  std::iota(this_indices.begin(), this_indices.end(), 0);

  std::vector<size_t> other_indices(other.n_terms);
  std::iota(other_indices.begin(), other_indices.end(), 0);

  // Sort the index vectors
  std::sort(this_indices.begin(), this_indices.end(), [&](size_t a, size_t b) {
    return std::lexicographical_compare(exponents.begin() + a * dimension,
                                        exponents.begin() + (a + 1) * dimension,
                                        exponents.begin() + b * dimension,
                                        exponents.begin() +
                                            (b + 1) * dimension);
  });

  std::sort(other_indices.begin(), other_indices.end(),
            [&](size_t a, size_t b) {
              return std::lexicographical_compare(
                  other.exponents.begin() + a * other.dimension,
                  other.exponents.begin() + (a + 1) * other.dimension,
                  other.exponents.begin() + b * other.dimension,
                  other.exponents.begin() + (b + 1) * other.dimension);
            });

  std::vector<int32_t> new_exponents;
  std::vector<std::complex<double>> new_coeffs;
  new_exponents.reserve(n_terms + other.n_terms);
  new_coeffs.reserve(n_terms + other.n_terms);

  size_t i = 0, j = 0;
  while (i < n_terms && j < other.n_terms) {
    auto this_exp_start = exponents.begin() + this_indices[i] * dimension;
    auto this_exp_end = this_exp_start + dimension;
    auto other_exp_start =
        other.exponents.begin() + other_indices[j] * other.dimension;
    auto other_exp_end = other_exp_start + other.dimension;

    if (std::lexicographical_compare(this_exp_start, this_exp_end,
                                     other_exp_start, other_exp_end)) {
      new_exponents.insert(new_exponents.end(), this_exp_start, this_exp_end);
      new_coeffs.push_back(coeffs[this_indices[i]]);
      i++;
    } else if (std::lexicographical_compare(other_exp_start, other_exp_end,
                                            this_exp_start, this_exp_end)) {
      new_exponents.insert(new_exponents.end(), other_exp_start, other_exp_end);
      new_coeffs.push_back(-other.coeffs[other_indices[j]]); // Subtract
      j++;
    } else {
      std::complex<double> diff =
          coeffs[this_indices[i]] - other.coeffs[other_indices[j]]; // Subtract
      if (std::abs(diff.real()) > 1e-16 || std::abs(diff.imag()) > 1e-16) {
        new_exponents.insert(new_exponents.end(), this_exp_start, this_exp_end);
        new_coeffs.push_back(diff);
      }
      i++;
      j++;
    }
  }

  while (i < n_terms) {
    auto this_exp_start = exponents.begin() + this_indices[i] * dimension;
    new_exponents.insert(new_exponents.end(), this_exp_start,
                         this_exp_start + dimension);
    new_coeffs.push_back(coeffs[this_indices[i]]);
    i++;
  }
  while (j < other.n_terms) {
    auto other_exp_start =
        other.exponents.begin() + other_indices[j] * other.dimension;
    new_exponents.insert(new_exponents.end(), other_exp_start,
                         other_exp_start + other.dimension);
    new_coeffs.push_back(-other.coeffs[other_indices[j]]); // Subtract
    j++;
  }

  exponents.swap(new_exponents);
  coeffs.swap(new_coeffs);
  n_terms = coeffs.size();
}

void MtfData::multiply_inplace(const MtfData &other) {
  if (dimension != other.dimension) {
    throw std::runtime_error("Dimensions must match for multiplication.");
  }

  if (n_terms == 0 || other.n_terms == 0) {
    clear();
    return;
  }

  std::vector<int32_t> new_exponents;
  std::vector<std::complex<double>> new_coeffs;
  size_t new_capacity = n_terms * other.n_terms;
  new_exponents.reserve(new_capacity * dimension);
  new_coeffs.reserve(new_capacity);

  for (size_t i = 0; i < n_terms; ++i) {
    for (size_t j = 0; j < other.n_terms; ++j) {
      std::vector<int32_t> temp_exp(dimension);
      for (int k = 0; k < dimension; ++k) {
        temp_exp[k] = exponents[i * dimension + k] +
                      other.exponents[j * other.dimension + k];
      }
      new_exponents.insert(new_exponents.end(), temp_exp.begin(),
                           temp_exp.end());
      new_coeffs.push_back(coeffs[i] * other.coeffs[j]);
    }
  }

  size_t new_n_terms = new_coeffs.size();
  std::vector<size_t> indices(new_n_terms);
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    return std::lexicographical_compare(
        new_exponents.begin() + a * dimension,
        new_exponents.begin() + (a + 1) * dimension,
        new_exponents.begin() + b * dimension,
        new_exponents.begin() + (b + 1) * dimension);
  });

  std::vector<int32_t> final_exponents;
  std::vector<std::complex<double>> final_coeffs;
  if (new_n_terms > 0) {
    final_exponents.reserve(new_n_terms * dimension);
    final_coeffs.reserve(new_n_terms);

    final_exponents.insert(
        final_exponents.end(), new_exponents.begin() + indices[0] * dimension,
        new_exponents.begin() + (indices[0] + 1) * dimension);
    final_coeffs.push_back(new_coeffs[indices[0]]);

    for (size_t i = 1; i < new_n_terms; ++i) {
      auto current_exp_start = new_exponents.begin() + indices[i] * dimension;
      auto last_final_exp_start = final_exponents.end() - dimension;

      if (std::equal(current_exp_start, current_exp_start + dimension,
                     last_final_exp_start)) {
        final_coeffs.back() += new_coeffs[indices[i]];
      } else {
        final_exponents.insert(final_exponents.end(), current_exp_start,
                               current_exp_start + dimension);
        final_coeffs.push_back(new_coeffs[indices[i]]);
      }
    }
  }

  exponents.swap(final_exponents);
  coeffs.swap(final_coeffs);
  n_terms = coeffs.size();

  // Prune near-zero coefficients
  size_t current_pos = 0;
  for (size_t i = 0; i < n_terms; ++i) {
    if (std::abs(coeffs[i].real()) > 1e-16 ||
        std::abs(coeffs[i].imag()) > 1e-16) {
      if (current_pos != i) {
        coeffs[current_pos] = coeffs[i];
        std::copy(exponents.begin() + i * dimension,
                  exponents.begin() + (i + 1) * dimension,
                  exponents.begin() + current_pos * dimension);
      }
      current_pos++;
    }
  }
  coeffs.resize(current_pos);
  exponents.resize(current_pos * dimension);
  n_terms = current_pos;
}

void MtfData::power_inplace(double exponent) {
  if (exponent == -0.5) {
    // Find constant term
    std::complex<double> const_term = {0.0, 0.0};
    int const_term_idx = -1;
    std::vector<int32_t> zero_exp(dimension, 0);
    for (size_t i = 0; i < n_terms; ++i) {
      if (std::equal(exponents.begin() + i * dimension,
                     exponents.begin() + (i + 1) * dimension,
                     zero_exp.begin())) {
        const_term = coeffs[i];
        const_term_idx = i;
        break;
      }
    }

    if (const_term_idx == -1 || (std::abs(const_term.real()) < 1e-16 &&
                                 std::abs(const_term.imag()) < 1e-16)) {
      throw std::runtime_error(
          "Cannot take isqrt of MTF with zero or non-existent constant term.");
    }

    // Rescale the polynomial
    std::complex<double> const_factor_isqrt = 1.0 / sqrt(const_term);
    for (auto &c : coeffs) {
      c /= const_term;
    }

    // x = (rescaled_poly - 1)
    MtfData x(dimension);
    x.reserve(n_terms > 0 ? n_terms - 1 : 0);
    for (size_t i = 0; i < n_terms; ++i) {
      if (i == (size_t)const_term_idx)
        continue;
      x.exponents.insert(x.exponents.end(), exponents.begin() + i * dimension,
                         exponents.begin() + (i + 1) * dimension);
      x.coeffs.push_back(coeffs[i]);
    }
    x.n_terms = x.coeffs.size();

    // Taylor series for isqrt(1+x) = sum(c_n * x^n)
    const auto &taylor_coeffs =
        mtf_coeffs::precomputed_coefficients.at("isqrt");
    MtfData result(dimension);
    result.exponents.insert(result.exponents.end(), zero_exp.begin(),
                            zero_exp.end());
    result.coeffs.push_back({taylor_coeffs[0], 0.0});
    result.n_terms = 1;

    MtfData x_power_n = x;

    for (size_t n = 1; n < taylor_coeffs.size(); ++n) {
      MtfData term = x_power_n;
      for (auto &c : term.coeffs) {
        c *= taylor_coeffs[n];
      }
      result.add_inplace(term);
      if (n < taylor_coeffs.size() - 1) {
        x_power_n.multiply_inplace(x);
      }
    }

    // Rescale result
    for (auto &c : result.coeffs) {
      c *= const_factor_isqrt;
    }

    // Swap with this
    exponents.swap(result.exponents);
    coeffs.swap(result.coeffs);
    n_terms = result.n_terms;

  } else {
    throw std::runtime_error(
        "Power function not implemented for this exponent.");
  }
}

MtfData MtfData::add(const MtfData &other) const {
  MtfData result = *this;
  result.add_inplace(other);
  return result;
}

MtfData MtfData::multiply(const MtfData &other) const {
  MtfData result = *this;
  result.multiply_inplace(other);
  return result;
}

MtfData MtfData::subtract(const MtfData &other) const {
  MtfData result = *this;
  result.add_inplace(other.negate());
  return result;
}

MtfData MtfData::negate() const {
  MtfData result = *this;
  for (auto &c : result.coeffs) {
    c = -c;
  }
  return result;
}
