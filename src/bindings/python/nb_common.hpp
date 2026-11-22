#pragma once

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::py {

namespace nb = nanobind;

/// Outputs are contiguous arrays or tuples of arrays. Test byte ranges so
/// differently typed views of the same allocation are covered too.
inline bool OverlapsOutput(const void* data, size_t bytes, nb::handle out) {
  if (!out.is_valid() || out.is_none() || bytes == 0) {
    return false;
  }
  if (nb::isinstance<nb::tuple>(out) || nb::isinstance<nb::list>(out)) {
    for (nb::handle item : nb::borrow<nb::sequence>(out)) {
      if (OverlapsOutput(data, bytes, item)) {
        return true;
      }
    }
    return false;
  }
  nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> target;
  if (!nb::try_cast(out, target, false) || target.nbytes() == 0) {
    return false; // invalid outputs are diagnosed by OutArray
  }
  const auto first = reinterpret_cast<uintptr_t>(data);
  const auto second = reinterpret_cast<uintptr_t>(target.data());
  return first < second + target.nbytes() && second < first + bytes;
}

/// Read-only C-contiguous view of a Python array-like as element type T.
/// An input that already is a C-contiguous array of T is borrowed as is, with
/// no copy and no call into Python. Anything else goes through
/// numpy.ascontiguousarray(obj, dtype_name). keepalive holds the borrowed
/// input or the converted array.
template<typename T>
struct ContiguousAdapter {
  nb::object keepalive;
  nb::ndarray<const T, nb::c_contig> array;

  ContiguousAdapter() = default;

  ContiguousAdapter(nb::handle obj, const char* dtype_name) {
    if (nb::try_cast(obj, array, /*convert=*/false)) {
      keepalive = nb::borrow(obj);
      return;
    }
    auto np = nb::module_::import_("numpy");
    keepalive = np.attr("ascontiguousarray")(obj, np.attr(dtype_name));
    array = nb::cast<nb::ndarray<const T, nb::c_contig>>(keepalive);
  }

  const T* data() const { return array.data(); }
  size_t size() const { return array.size(); }
  size_t ndim() const { return array.ndim(); }
  size_t shape(size_t i) const { return array.shape(i); }

  /// Preserve an input only when a caller-provided output overlaps it.
  void ProtectOutput(nb::handle out) {
    if (OverlapsOutput(data(), size() * sizeof(T), out)) {
      keepalive = nb::module_::import_("numpy").attr(
          "array")(keepalive, nb::arg("copy") = true);
      array = nb::cast<nb::ndarray<const T, nb::c_contig>>(keepalive);
    }
  }
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

/// numpy ufunc style output. With `out` given, results are written into that
/// array in place and it is returned (it has to be a writable C-contiguous
/// array of element type T and exactly the expected shape; nothing is converted
/// or copied behind the caller's back). Without `out`, a fresh array is
/// allocated.
template<typename T>
class OutArray {
public:
  OutArray(const nb::object& out, std::initializer_list<size_t> shape)
      : OutArray(out, {shape.begin(), shape.size()}) {}

  OutArray(const nb::object& out, splinepy::core::Span<const size_t> shape)
      : size_(ArraySize(shape)) {

    if (!out.is_none()) {
      nb::ndarray<T, nb::c_contig, nb::device::cpu> array;
      if (!nb::try_cast(out, array, /*convert=*/false)) {
        throw std::invalid_argument("out must be a writable, C-contiguous "
                                    "numpy array of the dtype of the result.");
      }
      bool matches = array.ndim() == shape.size();
      for (size_t i = 0; matches && i < shape.size(); ++i) {
        matches = array.shape(i) == shape[i];
      }
      if (!matches) {
        std::string expected;
        for (size_t i = 0; i < shape.size(); ++i) {
          expected += (i ? ", " : "") + std::to_string(shape[i]);
        }
        throw std::invalid_argument("out has the wrong shape, expected ("
                                    + expected + ").");
      }
      ptr_ = array.data();
      result_ = out;
      return;
    }

    auto result = AllocateNumPyArray<T>(shape);
    ptr_ = result.data();
    result_ = nb::cast(result);
  }

  T* data() const { return ptr_; }
  size_t size() const { return size_; }
  nb::object result() const { return result_; }

private:
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
  nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> first;
  if (nb::try_cast(out[0], first, false)
      && OverlapsOutput(first.data(), first.nbytes(), out[1])) {
    throw std::invalid_argument("Output arrays must not overlap each other.");
  }
  return {out[0], out[1]};
}

/// The arrays of a spline and the view into them. Copying one is how a call
/// pins a spline: a few references, no allocation, and the view stays valid
/// while the GIL is released, even if the spline replaces its arrays.
/// Absent arrays are null objects. A result made in C++ keeps its data in
/// owned and gets NumPy arrays (views of that data) only once Python asks.
struct SplineArrays {
  nb::object degrees;
  nb::object knot_vectors; // tuple of float64 arrays
  nb::object control_points;
  nb::object weights;
  std::shared_ptr<const splinepy::core::SplineData> owned;
  splinepy::core::SplineView view;
};

/// The native part of every splinepy Spline (bound as _core.SplineCore,
/// which the Python spline classes subclass): fixed slots for the arrays and
/// the SplineView into them, so a kernel reads a spline with no lookups.
struct NativeSpline {
  SplineArrays arrays;
  /// Bumped whenever degrees or knot vectors are replaced. Caches built from
  /// the layout compare it instead of the arrays.
  // Pack the validation state with the revision so spline instances keep
  // their existing allocation size. A query snapshots the bit with its arrays.
  uint64_t layout : 63;
  uint64_t layout_validated : 1;

  NativeSpline() : layout(0), layout_validated(0) {}

  /// Replaces the arrays. Each is converted to a C-contiguous array of its
  /// dtype if needed; None removes one. With validate, a complete spline is
  /// checked first and nothing changes if it is invalid.
  void Assign(nb::handle degrees,
              nb::handle knot_vectors,
              nb::handle control_points,
              nb::handle weights,
              bool validate);

  /// Takes over data made in C++, without creating NumPy arrays.
  void Adopt(splinepy::core::SplineData&& data);

  /// The arrays, created from owned data on first access.
  const SplineArrays& Arrays();
};

/// The native part of a Python spline; raises TypeError for other objects.
inline NativeSpline& NativeOf(nb::handle spline) {
  NativeSpline* native = nullptr;
  if (!nb::try_cast(spline, native) || native == nullptr) {
    throw nb::type_error("Expected a splinepy spline.");
  }
  return *native;
}

/// The arrays of a Python spline, pinned for the duration of a call.
inline SplineArrays Pin(nb::handle spline) { return NativeOf(spline).arrays; }

/// Pinned views of a Python list of splines, as a core batch.
struct ViewList {
  splinepy::core::Buffer<SplineArrays> pins;
  splinepy::core::Buffer<splinepy::core::SplineView> views;

  explicit ViewList(nb::handle splines)
      : pins(nb::len(splines)),
        views(pins.size()) {
    size_t i = 0;
    for (nb::handle h : splines) {
      pins[i] = Pin(h);
      views[i] = pins[i].view;
      ++i;
    }
  }

  splinepy::core::SplineBatchView Batch() const { return views.cspan(); }
  size_t size() const { return views.size(); }
  const splinepy::core::SplineView& operator[](size_t i) const {
    return views[i];
  }
};

} // namespace splinepy::py
