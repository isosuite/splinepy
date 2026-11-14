#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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

template<typename T, typename Storage, typename Deleter>
nb::ndarray<nb::numpy, T>
WrapOwnedArray(Storage& storage,
               splinepy::core::Span<const size_t> shape,
               Deleter deleter) {
  auto* data = storage.release();
  nb::capsule owner;
  try {
    owner = nb::capsule(data, deleter);
  } catch (...) {
    deleter(data);
    throw;
  }
  return {data, shape.size(), shape.data(), owner};
}

template<typename T>
nb::ndarray<nb::numpy, T>
WrapBuffer(splinepy::core::Buffer<std::remove_const_t<T>>& storage,
           splinepy::core::Span<const size_t> shape) {
  return WrapOwnedArray<T>(storage, shape, [](void* pointer) noexcept {
    splinepy::core::detail::AlignedFree(pointer);
  });
}

inline size_t ArraySize(splinepy::core::Span<const size_t> shape) {
  size_t size = 1;
  for (const auto extent : shape) {
    if (extent != 0 && size > std::numeric_limits<size_t>::max() / extent) {
      throw std::overflow_error("NumPy output shape is too large.");
    }
    size *= extent;
  }
  return size;
}

template<typename T>
nb::ndarray<nb::numpy, T>
AllocateNumPyArray(splinepy::core::Span<const size_t> shape) {
  std::unique_ptr<T[]> storage(new T[ArraySize(shape)]);
  return WrapOwnedArray<T>(storage, shape, [](void* pointer) noexcept {
    delete[] static_cast<T*>(pointer);
  });
}

template<typename T>
nb::object TakeNumPyBuffer(splinepy::core::Buffer<T>& buffer,
                           std::initializer_list<size_t> shape) {
  const splinepy::core::Span<const size_t> extents(shape.begin(), shape.size());
  // Empty Buffers have a null pointer, which cannot be owned by a capsule.
  return buffer.size() == 0 ? nb::cast(AllocateNumPyArray<T>(extents))
                            : nb::cast(WrapBuffer<T>(buffer, extents));
}

template<typename T>
nb::object CopyNumPyArray(const T* data,
                          splinepy::core::Span<const size_t> shape) {
  auto result = AllocateNumPyArray<T>(shape);
  if (result.size() != 0) {
    std::copy_n(data, result.size(), result.data());
  }
  return nb::cast(result);
}

template<typename T>
nb::object MakeNumPyArray1D(splinepy::core::Span<const T> span) {
  const size_t shape[] = {span.size()};
  return CopyNumPyArray(span.data(), {shape, 1});
}

inline nb::object MakeNumPyArray1D(const std::vector<bool>& vec) {
  const size_t shape[] = {vec.size()};
  auto result = AllocateNumPyArray<bool>({shape, 1});
  std::copy(vec.begin(), vec.end(), result.data());
  return nb::cast(result);
}

template<typename T>
nb::object MakeNumPyArray2D(const T* data, size_t rows, size_t cols) {
  const size_t shape[] = {rows, cols};
  return CopyNumPyArray(data, {shape, 2});
}

template<typename T>
nb::object MakeNumPyArray3D(const T* data, size_t d0, size_t d1, size_t d2) {
  const size_t shape[] = {d0, d1, d2};
  return CopyNumPyArray(data, {shape, 3});
}

/// numpy ufunc style output. With `out` given, results are written into that
/// array in place and it is returned (it has to be a writable C-contiguous
/// array of element type T and exactly the expected shape; nothing is converted
/// or copied behind the caller's back). Without `out`, a fresh array is
/// allocated.
template<typename T>
class OutArray {
public:
  OutArray(const nb::object& out, std::initializer_list<size_t> shape)
      : shape_(shape) {
    size_ = ArraySize(shape_);

    if (!out.is_none()) {
      nb::ndarray<T, nb::c_contig, nb::device::cpu> array;
      if (!nb::try_cast(out, array, /*convert=*/false)) {
        throw std::invalid_argument("out must be a writable, C-contiguous "
                                    "numpy array of the dtype of the result.");
      }
      bool matches = array.ndim() == shape_.size();
      for (size_t i = 0; matches && i < shape_.size(); ++i) {
        matches = array.shape(i) == shape_[i];
      }
      if (!matches) {
        std::string expected;
        for (size_t i = 0; i < shape_.size(); ++i) {
          expected += (i ? ", " : "") + std::to_string(shape_[i]);
        }
        throw std::invalid_argument("out has the wrong shape, expected ("
                                    + expected + ").");
      }
      ptr_ = array.data();
      result_ = out;
      return;
    }

    auto result = AllocateNumPyArray<T>(shape_);
    ptr_ = result.data();
    result_ = nb::cast(result);
  }

  T* data() const { return ptr_; }
  size_t size() const { return size_; }
  nb::object result() const { return result_; }

private:
  std::vector<size_t> shape_;
  size_t size_{1};
  T* ptr_{nullptr};
  nb::object result_;
};

/// out for functions with two results (e.g. basis and support): None or a
/// sequence of two arrays.
inline std::pair<nb::object, nb::object> SplitOutPair(const nb::object& out) {
  if (out.is_none()) {
    return {nb::none(), nb::none()};
  }
  if (!nb::isinstance<nb::sequence>(out) || nb::len(out) != 2) {
    throw std::invalid_argument("out must be None or a tuple of two arrays.");
  }
  return {out[0], out[1]};
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
