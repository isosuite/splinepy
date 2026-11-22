#include "splinepy/core/manipulation/insert_knots.hpp"

#include "splinepy/core/manipulation/axis_map.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

SplineData InsertKnots(const SplineView& spline,
                       int32_t axis,
                       Span<const double> knots_to_insert,
                       const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
  }
  Options apply = options;
  apply.validate = false;
  if (knots_to_insert.empty()) {
    return ApplyAxisMaps(spline, {}, true, apply);
  }
  if (axis < 0 || axis >= spline.para_dim) {
    throw SplinepyException("InsertKnots: axis out of range [0, para_dim).",
                            Status::kOutOfBounds);
  }
  const AxisKnots knots(spline, axis);
  const AxisMap map =
      KnotInsertionMap(knots, spline.degrees[axis], knots_to_insert);
  const AxisStep step{axis, &map};
  return ApplyAxisMaps(spline, {&step, 1}, true, apply);
}

SplineBatchData InsertKnots(SplineBatchView batch,
                            int32_t axis,
                            Span<const double> knots_to_insert,
                            const Options& options) {
  return detail::ForEachSpline(
      batch,
      options,
      [&](const SplineView& spline, const Options& worker) {
        return InsertKnots(spline, axis, knots_to_insert, worker);
      });
}

SplineBatchData UniformRefine(SplineBatchView batch,
                              int32_t knots_per_span,
                              const Options& options) {
  if (knots_per_span < 0) {
    throw SplinepyException("Knots per span must be non-negative.");
  }
  return detail::ForEachSpline(
      batch,
      options,
      [&](const SplineView& spline, const Options& worker) {
        if (options.validate) {
          ThrowIfInvalid(spline, options);
        }
        if (knots_per_span > 0 && !spline.HasKnotVectors()) {
          throw SplinepyException(
              "Uniform refinement requires patches with knot vectors.");
        }
        AxisMap maps[kMaxParaDim];
        AxisStep steps[kMaxParaDim];
        DynamicBuffer<double> knots;
        const int32_t n_steps = knots_per_span > 0 ? spline.para_dim : 0;
        for (int32_t axis = 0; axis < n_steps; ++axis) {
          knots.clear();
          const auto current = spline.knot_vectors[axis];
          const auto degree = static_cast<std::size_t>(spline.degrees[axis]);
          for (auto span = degree; span + 1 < current.size() - degree; ++span) {
            const double lower = current[span];
            const double upper = current[span + 1];
            if (lower == upper) {
              continue;
            }
            double previous = lower;
            for (int32_t i = 0; i < knots_per_span; ++i) {
              const double fraction = (i + 1.) / (knots_per_span + 1.);
              const double knot = (1. - fraction) * lower + fraction * upper;
              if (!(previous < knot && knot < upper)) {
                throw SplinepyException(
                    "Knot span is too narrow for the requested subdivision.");
              }
              knots.push_back(knot);
              previous = knot;
            }
          }
          maps[axis] =
              KnotInsertionMap(current, spline.degrees[axis], knots.cspan());
          steps[axis] = {axis, &maps[axis]};
        }
        Options apply = worker;
        apply.validate = false;
        return ApplyAxisMaps(spline,
                             {steps, static_cast<std::size_t>(n_steps)},
                             spline.HasKnotVectors(),
                             apply);
      });
}

} // namespace splinepy::core
