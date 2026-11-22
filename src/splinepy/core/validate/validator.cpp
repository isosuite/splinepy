#include "splinepy/core/validate/validator.hpp"

#include <cmath>
#include <sstream>

#include "splinepy/core/math/bspline_basis.hpp"
#include "splinepy/core/model/grid.hpp"

namespace splinepy::core {

namespace {

bool IsFinite(double val) noexcept { return std::isfinite(val); }

} // namespace

Status Validate(const SplineView& view, const Options& options) {
  if (view.para_dim <= 0 || view.dim <= 0) {
    return Status::kInvalidArgument;
  }

  if (view.degrees.size() != static_cast<std::size_t>(view.para_dim)) {
    return Status::kDimensionMismatch;
  }

  for (int32_t i = 0; i < view.para_dim; ++i) {
    if (view.degrees[i] < 0) {
      return Status::kInvalidArgument;
    }
  }

  if (view.HasKnotVectors()) {
    if (view.knot_vectors.size() != static_cast<std::size_t>(view.para_dim)) {
      return Status::kDimensionMismatch;
    }
    for (int32_t i = 0; i < view.para_dim; ++i) {
      const auto& kv = view.knot_vectors[i];
      const int32_t deg = view.degrees[i];
      const int64_t min_knots = 2 * (static_cast<int64_t>(deg) + 1);
      if (static_cast<int64_t>(kv.size()) < min_knots) {
        return Status::kInvalidArgument;
      }
      for (std::size_t j = 0; j < kv.size(); ++j) {
        if (!IsFinite(kv[j])) {
          return Status::kInvalidArgument;
        }
        if (j > 0 && kv[j] < kv[j - 1] - options.tolerance) {
          return Status::kInvalidArgument;
        }
      }
    }
  }

  if (!view.control_points_data) {
    return Status::kInvalidArgument;
  }

  const int64_t expected_cps = view.ExpectedTotalControlPoints();
  if (view.num_control_points != expected_cps || expected_cps <= 0) {
    return Status::kDimensionMismatch;
  }

  for (int64_t i = 0; i < view.num_control_points; ++i) {
    for (int32_t d = 0; d < view.dim; ++d) {
      if (!IsFinite(view.Point(i, d))) {
        return Status::kInvalidArgument;
      }
    }
  }

  if (view.IsRational()) {
    if (!view.weights_data) {
      return Status::kInvalidArgument;
    }
    for (int64_t i = 0; i < view.num_control_points; ++i) {
      const double w = view.Weight(i);
      if (!IsFinite(w) || w <= 0.0) {
        return Status::kInvalidArgument;
      }
    }
  }

  return Status::kOk;
}

Status Validate(SplineBatchView batch, const Options& options) {
  for (std::size_t i = 0; i < batch.size(); ++i) {
    Status s = Validate(batch[i], options);
    if (s != Status::kOk) {
      return s;
    }
  }
  return Status::kOk;
}

void ThrowIfInvalid(const SplineView& view, const Options& options) {
  if (view.para_dim <= 0) {
    throw SplinepyException("Parametric dimension must be > 0",
                            Status::kInvalidArgument);
  }
  if (view.dim <= 0) {
    throw SplinepyException("Physical dimension must be > 0",
                            Status::kInvalidArgument);
  }
  if (view.degrees.size() != static_cast<std::size_t>(view.para_dim)) {
    std::ostringstream oss;
    oss << "Degrees size (" << view.degrees.size()
        << ") does not match parametric dimension (" << view.para_dim << ")";
    throw SplinepyException(oss.str(), Status::kDimensionMismatch);
  }
  for (int32_t i = 0; i < view.para_dim; ++i) {
    if (view.degrees[i] < 0) {
      std::ostringstream oss;
      oss << "Degree at axis " << i << " (" << view.degrees[i]
          << ") must be non-negative";
      throw SplinepyException(oss.str(), Status::kInvalidArgument);
    }
  }
  if (view.HasKnotVectors()) {
    if (view.knot_vectors.size() != static_cast<std::size_t>(view.para_dim)) {
      std::ostringstream oss;
      oss << "Knot vectors count (" << view.knot_vectors.size()
          << ") does not match parametric dimension (" << view.para_dim << ")";
      throw SplinepyException(oss.str(), Status::kDimensionMismatch);
    }
    for (int32_t i = 0; i < view.para_dim; ++i) {
      const auto& kv = view.knot_vectors[i];
      const int32_t deg = view.degrees[i];
      const int64_t min_knots = 2 * (static_cast<int64_t>(deg) + 1);
      if (static_cast<int64_t>(kv.size()) < min_knots) {
        std::ostringstream oss;
        oss << "Knot vector at axis " << i << " has size " << kv.size()
            << ", but must have at least 2*(degree+1) = " << min_knots
            << " knots";
        throw SplinepyException(oss.str(), Status::kInvalidArgument);
      }
      for (std::size_t j = 0; j < kv.size(); ++j) {
        if (!IsFinite(kv[j])) {
          std::ostringstream oss;
          oss << "Non-finite knot value at axis " << i << ", index " << j;
          throw SplinepyException(oss.str(), Status::kInvalidArgument);
        }
        if (j > 0 && kv[j] < kv[j - 1] - options.tolerance) {
          std::ostringstream oss;
          oss << "Knot vector at axis " << i << " is not non-decreasing: knot["
              << (j - 1) << "]=" << kv[j - 1] << " > knot[" << j
              << "]=" << kv[j];
          throw SplinepyException(oss.str(), Status::kInvalidArgument);
        }
      }
    }
  }

  if (!view.control_points_data) {
    throw SplinepyException("Control points data pointer is null",
                            Status::kInvalidArgument);
  }

  const int64_t expected_cps = view.ExpectedTotalControlPoints();
  if (view.num_control_points != expected_cps) {
    std::ostringstream oss;
    oss << "Control points count (" << view.num_control_points
        << ") does not match expected count (" << expected_cps << ")";
    throw SplinepyException(oss.str(), Status::kDimensionMismatch);
  }

  for (int64_t i = 0; i < view.num_control_points; ++i) {
    for (int32_t d = 0; d < view.dim; ++d) {
      if (!IsFinite(view.Point(i, d))) {
        std::ostringstream oss;
        oss << "Control point " << i << " component " << d << " is non-finite";
        throw SplinepyException(oss.str(), Status::kInvalidArgument);
      }
    }
  }

  if (view.IsRational()) {
    if (!view.weights_data) {
      throw SplinepyException("Rational spline has null weights pointer",
                              Status::kInvalidArgument);
    }
    for (int64_t i = 0; i < view.num_control_points; ++i) {
      const double w = view.Weight(i);
      if (!IsFinite(w)) {
        std::ostringstream oss;
        oss << "Weight at index " << i << " is non-finite";
        throw SplinepyException(oss.str(), Status::kInvalidArgument);
      }
      if (w <= 0.0) {
        std::ostringstream oss;
        oss << "Weight at index " << i << " (" << w
            << ") must be strictly positive";
        throw SplinepyException(oss.str(), Status::kInvalidArgument);
      }
    }
  }
}

void ThrowIfInvalidQueryValues(const SplineView& view,
                               Span<const double> queries) {
  const auto check = [&](int64_t cp) {
    for (int32_t c = 0; c < view.dim; ++c) {
      if (!std::isfinite(view.Point(cp, c))) {
        throw SplinepyException("A queried control point is non-finite.");
      }
    }
    if (view.IsRational()) {
      const double weight = view.Weight(cp);
      if (!(weight > 0.) || !std::isfinite(weight)) {
        throw SplinepyException(
            "A queried weight must be finite and positive.");
      }
    }
  };
  const auto n = queries.size() / view.para_dim;
  if (n == 0) {
    return;
  }
  const auto supports = view.NumSupports();
  if (n >= static_cast<std::size_t>((view.num_control_points - 1) / supports
                                    + 1)) {
    for (int64_t cp = 0; cp < view.num_control_points; ++cp) {
      check(cp);
    }
    return;
  }
  const auto grid = ControlGrid(view);
  AxisArray<int32_t> starts;
  starts.resize(view.para_dim);
  for (std::size_t q = 0; q < n; ++q) {
    for (int32_t d = 0; d < view.para_dim; ++d) {
      starts[d] = view.HasKnotVectors()
                      ? FindSpan(static_cast<int32_t>(grid.n[d]),
                                 view.degrees[d],
                                 queries[q * view.para_dim + d],
                                 view.knot_vectors[d])
                            - view.degrees[d]
                      : 0;
    }
    for (int64_t support = 0; support < supports; ++support) {
      int64_t cp = 0, remainder = support;
      for (int32_t d = 0; d < view.para_dim; ++d) {
        const int64_t width = static_cast<int64_t>(view.degrees[d]) + 1;
        cp += (starts[d] + remainder % width) * grid.stride[d];
        remainder /= width;
      }
      check(cp);
    }
  }
}

void ThrowIfInvalid(SplineBatchView batch, const Options& options) {
  for (std::size_t i = 0; i < batch.size(); ++i) {
    try {
      ThrowIfInvalid(batch[i], options);
    } catch (const SplinepyException& ex) {
      std::ostringstream oss;
      oss << "Error in spline batch at index " << i << ": " << ex.what();
      throw SplinepyException(oss.str(), ex.status());
    }
  }
}

} // namespace splinepy::core
