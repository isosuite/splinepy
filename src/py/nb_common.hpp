#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::py {

namespace nb = nanobind;

template<typename T>
struct ContiguousAdapter {
  nb::object keepalive;
  nb::ndarray<const T, nb::c_contig> array;

  explicit ContiguousAdapter(nb::handle obj, const char* dtype_name) {
    auto np = nb::module_::import_("numpy");
    keepalive = np.attr("ascontiguousarray")(obj, np.attr(dtype_name));
    array = nb::cast<nb::ndarray<const T, nb::c_contig>>(keepalive);
  }

  const T* data() const { return array.data(); }
  size_t size() const { return array.size(); }
  size_t ndim() const { return array.ndim(); }
  size_t shape(size_t i) const { return array.shape(i); }
};

template<typename T>
nb::object MakeNumPyArray1D(splinepy::core::Span<const T> span) {
  const size_t size = span.size();
  size_t shape[1] = {size};
  T* ptr = new T[size];
  std::copy_n(span.data(), size, ptr);
  nb::capsule owner(ptr, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::cast(
      nb::ndarray<nb::numpy, T, nb::shape<1>>(ptr, 1, shape, owner));
}

inline nb::object MakeNumPyArray1D(const std::vector<bool>& vec) {
  const size_t size = vec.size();
  size_t shape[1] = {size};
  bool* ptr = new bool[size];
  for (size_t i = 0; i < size; ++i)
    ptr[i] = vec[i];
  nb::capsule owner(ptr,
                    [](void* p) noexcept { delete[] static_cast<bool*>(p); });
  return nb::cast(
      nb::ndarray<nb::numpy, bool, nb::shape<1>>(ptr, 1, shape, owner));
}

template<typename T>
nb::object MakeNumPyArray2D(const T* data, size_t rows, size_t cols) {
  const size_t total = rows * cols;
  size_t shape[2] = {rows, cols};
  T* ptr = new T[total];
  std::copy_n(data, total, ptr);
  nb::capsule owner(ptr, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::cast(
      nb::ndarray<nb::numpy, T, nb::shape<2>>(ptr, 2, shape, owner));
}

template<typename T>
nb::object MakeNumPyArray3D(const T* data, size_t d0, size_t d1, size_t d2) {
  const size_t total = d0 * d1 * d2;
  size_t shape[3] = {d0, d1, d2};
  T* ptr = new T[total];
  std::copy_n(data, total, ptr);
  nb::capsule owner(ptr, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::cast(
      nb::ndarray<nb::numpy, T, nb::shape<3>>(ptr, 3, shape, owner));
}

/// Helper holding temporary arrays and spans to safely present a SplineView.
struct SplineViewWrapper {
  // Nanobind array objects to keep memory alive
  nb::object cps_keepalive;
  nb::object deg_keepalive;
  std::vector<nb::object> kv_keepalive;
  nb::object weights_keepalive;

  nb::ndarray<const double, nb::c_contig> cps_arr;
  nb::ndarray<const int32_t, nb::c_contig> degrees_arr;
  std::vector<nb::ndarray<const double, nb::c_contig>> kv_arrs;
  std::vector<splinepy::core::Span<const double>> kv_spans;
  std::optional<nb::ndarray<const double, nb::c_contig>> weights_arr;

  splinepy::core::SplineView view;

  SplineViewWrapper() = default;

  void Init(nb::handle cps_obj,
            nb::handle deg_obj,
            nb::handle kvs_obj,
            nb::handle weights_obj) {
    auto np = nb::module_::import_("numpy");

    if (!cps_obj.is_none()) {
      cps_keepalive = np.attr("ascontiguousarray")(cps_obj, np.attr("float64"));
      cps_arr =
          nb::cast<nb::ndarray<const double, nb::c_contig>>(cps_keepalive);
      view.control_points_data = cps_arr.data();
      view.num_control_points = static_cast<int64_t>(
          cps_arr.ndim() >= 2 ? cps_arr.shape(0) : cps_arr.size());
      view.dim = static_cast<int32_t>(
          cps_arr.ndim() >= 2 ? cps_arr.shape(cps_arr.ndim() - 1) : 1);
      view.cp_stride = view.dim;
      view.dim_stride = 1;
    }

    if (!deg_obj.is_none()) {
      deg_keepalive = np.attr("ascontiguousarray")(deg_obj, np.attr("int32"));
      degrees_arr =
          nb::cast<nb::ndarray<const int32_t, nb::c_contig>>(deg_keepalive);
      view.para_dim = static_cast<int32_t>(degrees_arr.size());
      view.degrees = splinepy::core::Span<const int32_t>(degrees_arr.data(),
                                                         degrees_arr.size());
    }

    if (!kvs_obj.is_none()) {
      nb::object seq_obj = nb::borrow(kvs_obj);
      if (nb::hasattr(kvs_obj, "knot_vectors")) {
        seq_obj = kvs_obj.attr("knot_vectors");
      }
      const size_t n_kvs = nb::len(seq_obj);
      kv_keepalive.reserve(n_kvs);
      kv_arrs.reserve(n_kvs);
      kv_spans.reserve(n_kvs);
      for (size_t i = 0; i < n_kvs; ++i) {
        nb::object item = seq_obj[nb::int_(i)];
        if (nb::hasattr(item, "numpy")) {
          item = item.attr("numpy")();
        }
        nb::object item_arr =
            np.attr("ascontiguousarray")(item, np.attr("float64"));
        kv_keepalive.push_back(item_arr);
        auto arr = nb::cast<nb::ndarray<const double, nb::c_contig>>(item_arr);
        kv_arrs.push_back(arr);
        kv_spans.emplace_back(arr.data(), arr.size());
      }
      if (!kv_spans.empty()) {
        view.knot_vectors =
            splinepy::core::Span<const splinepy::core::Span<const double>>(
                kv_spans.data(),
                kv_spans.size());
      }
    }

    if (!weights_obj.is_none()) {
      weights_keepalive =
          np.attr("ascontiguousarray")(weights_obj, np.attr("float64"));
      weights_arr =
          nb::cast<nb::ndarray<const double, nb::c_contig>>(weights_keepalive);
      view.weights_data = weights_arr->data();
      view.weight_stride = 1;
    } else {
      view.weights_data = nullptr;
    }
  }

  SplineViewWrapper(const SplineViewWrapper& other)
      : cps_keepalive(other.cps_keepalive),
        deg_keepalive(other.deg_keepalive),
        kv_keepalive(other.kv_keepalive),
        weights_keepalive(other.weights_keepalive),
        cps_arr(other.cps_arr),
        degrees_arr(other.degrees_arr),
        kv_arrs(other.kv_arrs),
        kv_spans(other.kv_spans),
        weights_arr(other.weights_arr),
        view(other.view) {
    if (!kv_spans.empty()) {
      view.knot_vectors =
          splinepy::core::Span<const splinepy::core::Span<const double>>(
              kv_spans.data(),
              kv_spans.size());
    }
  }

  SplineViewWrapper(SplineViewWrapper&& other) noexcept
      : cps_keepalive(std::move(other.cps_keepalive)),
        deg_keepalive(std::move(other.deg_keepalive)),
        kv_keepalive(std::move(other.kv_keepalive)),
        weights_keepalive(std::move(other.weights_keepalive)),
        cps_arr(std::move(other.cps_arr)),
        degrees_arr(std::move(other.degrees_arr)),
        kv_arrs(std::move(other.kv_arrs)),
        kv_spans(std::move(other.kv_spans)),
        weights_arr(std::move(other.weights_arr)),
        view(other.view) {
    if (!kv_spans.empty()) {
      view.knot_vectors =
          splinepy::core::Span<const splinepy::core::Span<const double>>(
              kv_spans.data(),
              kv_spans.size());
    }
  }

  SplineViewWrapper& operator=(const SplineViewWrapper& other) {
    if (this != &other) {
      cps_keepalive = other.cps_keepalive;
      deg_keepalive = other.deg_keepalive;
      kv_keepalive = other.kv_keepalive;
      weights_keepalive = other.weights_keepalive;
      cps_arr = other.cps_arr;
      degrees_arr = other.degrees_arr;
      kv_arrs = other.kv_arrs;
      kv_spans = other.kv_spans;
      weights_arr = other.weights_arr;
      view = other.view;
      if (!kv_spans.empty()) {
        view.knot_vectors =
            splinepy::core::Span<const splinepy::core::Span<const double>>(
                kv_spans.data(),
                kv_spans.size());
      }
    }
    return *this;
  }

  SplineViewWrapper& operator=(SplineViewWrapper&& other) noexcept {
    if (this != &other) {
      cps_keepalive = std::move(other.cps_keepalive);
      deg_keepalive = std::move(other.deg_keepalive);
      kv_keepalive = std::move(other.kv_keepalive);
      weights_keepalive = std::move(other.weights_keepalive);
      cps_arr = std::move(other.cps_arr);
      degrees_arr = std::move(other.degrees_arr);
      kv_arrs = std::move(other.kv_arrs);
      kv_spans = std::move(other.kv_spans);
      weights_arr = std::move(other.weights_arr);
      view = other.view;
      if (!kv_spans.empty()) {
        view.knot_vectors =
            splinepy::core::Span<const splinepy::core::Span<const double>>(
                kv_spans.data(),
                kv_spans.size());
      }
    }
    return *this;
  }

  SplineViewWrapper(nb::handle cps_obj,
                    nb::handle deg_obj,
                    nb::handle kvs_obj,
                    nb::handle weights_obj) {
    Init(cps_obj, deg_obj, kvs_obj, weights_obj);
  }

  explicit SplineViewWrapper(nb::handle spline_obj);
};

struct PySplineHolder {
  SplineViewWrapper wrapper;
  std::string name;
  const splinepy::core::SplineView& view;

  explicit PySplineHolder(nb::handle obj) : wrapper(obj), view(wrapper.view) {
    if (nb::hasattr(obj, "name")) {
      name = nb::cast<std::string>(obj.attr("name"));
    } else if (nb::hasattr(obj, "whatami")) {
      name = nb::cast<std::string>(obj.attr("whatami"));
    } else {
      name = "Unknown";
    }
  }
};

} // namespace splinepy::py
