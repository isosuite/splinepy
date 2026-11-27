#include "splinepy/core/validate/validator.hpp"

#include <cmath>
#include <limits>
#include <sstream>

#include "splinepy/core/math/bspline_basis.hpp"
#include "splinepy/core/model/grid.hpp"

namespace splinepy::core {

namespace {

bool IsFinite(double val) noexcept { return std::isfinite(val); }

Status ValidateImpl(const SplineView& view, bool throw_on_error) {
  auto fail = [&](Status status, auto&& build_msg) -> Status {
    if (throw_on_error) {
      std::ostringstream oss;
      build_msg(oss);
      throw SplinepyException(oss.str(), status);
    }
    return status;
  };

  if (view.para_dim <= 0) {
    return fail(Status::kInvalidArgument,
                [](auto& os) { os << "Parametric dimension must be > 0"; });
  }
  if (view.dim <= 0) {
    return fail(Status::kInvalidArgument,
                [](auto& os) { os << "Physical dimension must be > 0"; });
  }
  if (view.degrees.size() != static_cast<std::size_t>(view.para_dim)) {
    return fail(Status::kDimensionMismatch, [&](auto& os) {
      os << "Degrees size (" << view.degrees.size()
         << ") does not match parametric dimension (" << view.para_dim << ")";
    });
  }
  for (int32_t i = 0; i < view.para_dim; ++i) {
    if (view.degrees[i] < 0) {
      return fail(Status::kInvalidArgument, [&](auto& os) {
        os << "Degree at axis " << i << " (" << view.degrees[i]
           << ") must be non-negative";
      });
    }
  }
  if (!view.knot_vectors.empty()) {
    if (view.knot_vectors.size() != static_cast<std::size_t>(view.para_dim)) {
      return fail(Status::kDimensionMismatch, [&](auto& os) {
        os << "Knot vectors count (" << view.knot_vectors.size()
           << ") does not match parametric dimension (" << view.para_dim << ")";
      });
    }
    for (int32_t i = 0; i < view.para_dim; ++i) {
      const auto& kv = view.knot_vectors[i];
      const int32_t deg = view.degrees[i];
      const int64_t min_knots = 2 * (static_cast<int64_t>(deg) + 1);
      if (static_cast<int64_t>(kv.size()) < min_knots) {
        return fail(Status::kInvalidArgument, [&](auto& os) {
          os << "Knot vector at axis " << i << " has size " << kv.size()
             << ", but must have at least 2*(degree+1) = " << min_knots
             << " knots";
        });
      }
      for (std::size_t j = 0; j < kv.size(); ++j) {
        if (!IsFinite(kv[j])) {
          return fail(Status::kInvalidArgument, [&](auto& os) {
            os << "Non-finite knot value at axis " << i << ", index " << j;
          });
        }
        if (j > 0 && kv[j] < kv[j - 1]) {
          return fail(Status::kInvalidArgument, [&](auto& os) {
            os << "Knot vector at axis " << i << " is not non-decreasing: knot["
               << (j - 1) << "]=" << kv[j - 1] << " > knot[" << j
               << "]=" << kv[j];
          });
        }
      }
      const std::size_t n_cps = kv.size() - static_cast<std::size_t>(deg) - 1;
      if (!(kv[n_cps] > kv[deg])) {
        return fail(Status::kInvalidArgument, [&](auto& os) {
          os << "Knot vector at axis " << i << " has a degenerate domain";
        });
      }
    }
  }

  if (!view.control_points_data) {
    return fail(Status::kInvalidArgument,
                [](auto& os) { os << "Control points data pointer is null"; });
  }

  const int64_t expected_cps = view.ExpectedTotalControlPoints();
  if (view.num_control_points != expected_cps || expected_cps <= 0) {
    return fail(Status::kDimensionMismatch, [&](auto& os) {
      os << "Control points count (" << view.num_control_points
         << ") does not match expected count (" << expected_cps << ")";
    });
  }

  for (int64_t i = 0; i < view.num_control_points; ++i) {
    for (int32_t d = 0; d < view.dim; ++d) {
      if (!IsFinite(view.Point(i, d))) {
        return fail(Status::kInvalidArgument, [&](auto& os) {
          os << "Control point " << i << " component " << d << " is non-finite";
        });
      }
    }
  }

  if (view.IsRational()) {
    for (int64_t i = 0; i < view.num_control_points; ++i) {
      const double w = view.Weight(i);
      if (!IsFinite(w)) {
        return fail(Status::kInvalidArgument, [&](auto& os) {
          os << "Weight at index " << i << " is non-finite";
        });
      }
      // a subnormal weight has no finite reciprocal
      if (w < std::numeric_limits<double>::min()) {
        return fail(Status::kInvalidArgument, [&](auto& os) {
          os << "Weight at index " << i << " (" << w
             << ") must be a positive normal number";
        });
      }
    }
  }
  return Status::kOk;
}

} // namespace

Status Validate(const SplineView& view, const Options&) {
  return ValidateImpl(view, false);
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

void ThrowIfInvalid(const SplineView& view, const Options&) {
  ValidateImpl(view, true);
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
      if (!(weight >= std::numeric_limits<double>::min())
          || !std::isfinite(weight)) {
        throw SplinepyException(
            "A queried weight must be a finite, positive normal number.");
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
