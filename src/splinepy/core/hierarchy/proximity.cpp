#include "splinepy/core/hierarchy/proximity.hpp"

#include <array>
#include <cmath>
#include <limits>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/hierarchy/detail/grid.hpp"
#include "splinepy/core/hierarchy/export.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/proximity/proximity.hpp"

namespace splinepy::core {
namespace {

using detail::CheckedProduct;

class SamplingStep : public proximity::InitialGuess {
public:
  explicit SamplingStep(Span<const double> step) : step_(step) {}
  void Guess(const double*, double*) const override {}
  const double* StepSize() const override { return step_.data(); }

private:
  Span<const double> step_;
};

double SquaredDistance(Span<const double> a, Span<const double> b) {
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double difference = a[i] - b[i];
    result += difference * difference;
  }
  return result;
}

} // namespace

HierarchyProximityData BuildHierarchyProximity(const SplineView& spline,
                                               const ThbCacheData& extraction,
                                               const ThbGeometryData& geometry,
                                               const Options& options) {
  HierarchyProximityData result;
  result.bezier = BezierLeafGeometry(spline, extraction, geometry, options);
  for (const auto weight : result.bezier.weights) {
    if (!std::isfinite(weight) || !(weight > 0.0)) {
      throw SplinepyException(
          "Proximity requires positive finite Bezier weights");
    }
  }
  result.bvh = ElementBvh(spline.para_dim,
                          spline.dim,
                          extraction.local_size,
                          extraction.bounds,
                          result.bezier.points);
  if (!result.bvh.IsValid()) {
    throw SplinepyException("Could not build hierarchy proximity bounds");
  }
  return result;
}

HierarchyProximityResult
HierarchyProximities(const SplineView& spline,
                     const ThbCacheData& extraction,
                     const HierarchyProximityData& data,
                     Span<const double> queries,
                     Span<const int32_t> resolutions,
                     double tolerance,
                     int32_t max_iterations,
                     bool tight_bounds,
                     bool verbose,
                     const Options& options) {
  if (!data.bvh.IsValid() || spline.para_dim <= 0 || spline.dim <= 0
      || data.bvh.ParaDim() != spline.para_dim || data.bvh.Dim() != spline.dim
      || !std::isfinite(tolerance) || tolerance <= 0) {
    throw SplinepyException("Invalid hierarchy proximity options");
  }
  proximity::ValidateIterationBudget(max_iterations);
  const auto pd = static_cast<std::size_t>(spline.para_dim);
  const auto dim = static_cast<std::size_t>(spline.dim);
  if (queries.size() % dim || resolutions.size() != pd
      || std::any_of(resolutions.begin(),
                     resolutions.end(),
                     [](auto n) { return n < 2; })
      || std::any_of(queries.begin(), queries.end(), [](auto x) {
           return !std::isfinite(x);
         })) {
    throw SplinepyException(
        "Expected finite physical queries and resolutions of at least two");
  }

  const auto count = queries.size() / dim;
  HierarchyProximityResult result;
  result.parameters.resize(CheckedProduct(count, pd));
  if (count == 0) {
    return result;
  }
  const auto first_size = CheckedProduct(pd, dim);
  const auto second_size = CheckedProduct(pd, first_size);
  // The existing local solvers use int indices for derivative and SLSQP
  // workspace arrays. Check their largest workspace before entering them.
  if (second_size > INT32_MAX) {
    throw SplinepyException("Hierarchy proximity exceeds local solver capacity",
                            Status::kOutOfBounds);
  }
  const auto n1 = static_cast<int64_t>(pd) + 1;
  const auto slsqp_size = 3 * n1 * (n1 + 1) + (n1 + 1) * (2 * n1 + 2) + 4 * n1
                          + 3 * n1 * n1 + n1 + n1 * (n1 - 1) / 2 + 3 * (n1 - 1)
                          + 3 * n1 + 1;
  if (slsqp_size > INT32_MAX) {
    throw SplinepyException("Hierarchy proximity exceeds local solver capacity",
                            Status::kOutOfBounds);
  }
  if (verbose) {
    result.points.resize(queries.size());
    result.differences.resize(queries.size());
    result.distances.resize(count);
    result.residuals.resize(count);
    result.first_derivatives.resize(CheckedProduct(count, first_size));
    result.second_derivatives.resize(CheckedProduct(count, second_size));
  }

  // Keep the sampling budget bounded even for large resolutions or many
  // axes. Reduced axes are sampled at their midpoints if necessary.
  constexpr std::size_t sample_budget = 4096;
  Buffer<int32_t> samples(pd);
  Buffer<std::size_t> strides(pd);
  Buffer<double> step(pd);
  std::size_t sample_count = 1;
  for (std::size_t axis = 0; axis < pd; ++axis) {
    samples[axis] = std::min<std::size_t>(resolutions[axis], sample_budget);
  }
  for (;;) {
    sample_count = 1;
    for (auto n : samples) {
      sample_count *= n;
      if (sample_count > sample_budget) {
        break;
      }
    }
    if (sample_count <= sample_budget) {
      break;
    }
    auto largest = std::max_element(samples.begin(), samples.end());
    *largest = (*largest + 1) / 2;
    if (*largest == 2
        && std::none_of(samples.begin(), samples.end(), [](auto n) {
             return n > 2;
           })) {
      // All other axes have at most two samples. Reduce this axis to one.
      *largest = 1;
    }
  }
  Buffer<double> unit_queries(CheckedProduct(sample_count, pd));
  std::size_t stride = 1;
  for (std::size_t axis = 0; axis < pd; ++axis) {
    const auto n = samples[axis];
    strides[axis] = stride;
    step[axis] = n == 1 ? 1.0 : 1.0 / (n - 1);
    for (std::size_t i = 0; i < sample_count; ++i) {
      unit_queries[i * pd + axis] =
          n == 1 ? 0.5 : (i / stride % n) * step[axis];
    }
    stride *= n;
  }

  auto local_options = options;
  local_options.nthreads = 1;
  local_options.validate = false;
  local_options.workspace = nullptr;
  // Cache physical samples if they fit in the same budget as tensor proximity.
  Buffer<double> sampled;
  const auto sample_values = CheckedProduct(sample_count, dim);
  const auto leaves = static_cast<std::size_t>(data.bvh.NumElements());
  auto local_view = [&](std::size_t leaf) {
    auto view = spline;
    view.hierarchy = nullptr;
    view.knot_vectors = {};
    view.num_control_points = extraction.local_size;
    view.cp_stride = spline.dim;
    view.dim_stride = 1;
    view.weight_stride = 1;
    view.control_points_data =
        data.bezier.points.data() + leaf * extraction.local_size * dim;
    view.weights_data = spline.IsRational() ? data.bezier.weights.data()
                                                  + leaf * extraction.local_size
                                            : nullptr;
    return view;
  };
  if (count >= (leaves + 3) / 4 && count > 1
      && leaves <= (16u << 20) / sample_values) {
    sampled.resize(leaves * sample_values);
    ParallelFor(
        std::size_t{0},
        leaves,
        options.nthreads,
        [&](std::size_t leaf) {
          Evaluate(local_view(leaf),
                   unit_queries,
                   sampled.span().subspan(leaf * sample_values, sample_values),
                   local_options);
        });
  }

  const auto threads =
      std::min(count, static_cast<std::size_t>(std::max(1, options.nthreads)));
  // Separate query scratch from the workspaces of nested tensor kernels.
  Workspace default_workspace(MemoryArena::kDefaultBlockSize, 1);
  auto& workspace = options.workspace ? *options.workspace : default_workspace;
  workspace.GetThreadArena(static_cast<int>(threads - 1));
  ParallelForWithThreadId(
      std::size_t{0},
      count,
      options.nthreads,
      [&](std::size_t q, int thread) {
        auto& arena = workspace.GetThreadArena(thread);
        auto scope = arena.CreateScope();
        const auto query = queries.subspan(q * dim, dim);
        auto parameters = result.parameters.span().subspan(q * pd, pd);
        auto candidate = arena.AllocateSpan<double>(pd);
        auto mapped = arena.AllocateSpan<double>(pd);
        auto unit = arena.AllocateSpan<double>(pd);
        auto best_unit = arena.AllocateSpan<double>(pd);
        auto point = arena.AllocateSpan<double>(dim);
        auto difference = arena.AllocateSpan<double>(dim);
        auto distances = arena.AllocateSpan<double>(sample_count);
        auto physical_samples = sampled.empty()
                                    ? arena.AllocateSpan<double>(sample_values)
                                    : Span<double>();
        double best = std::numeric_limits<double>::infinity();
        double best_residual = best;
        std::size_t best_leaf = 0;
        bool found = false;

        data.bvh.Traverse(
            query.data(),
            [&](int64_t element, double) {
              const auto leaf = static_cast<std::size_t>(element);
              const auto view = local_view(leaf);
              proximity::Proximity solver(view);
              if (tight_bounds) {
                solver.SetInitialGuess(std::make_unique<SamplingStep>(step));
              }
              if (sampled.empty()) {
                Evaluate(view, unit_queries, physical_samples, local_options);
              }
              const auto values =
                  sampled.empty()
                      ? Span<const double>(physical_samples)
                      : sampled.cspan().subspan(leaf * sample_values,
                                                sample_values);
              for (std::size_t i = 0; i < sample_count; ++i) {
                distances[i] =
                    SquaredDistance(values.subspan(i * dim, dim), query);
              }

              // The four best sampled basins give independent starts on a
              // curved leaf.
              std::array<std::size_t, 4> starts;
              starts.fill(sample_count);
              for (std::size_t i = 0; i < sample_count; ++i) {
                bool minimum = true;
                for (std::size_t axis = 0; axis < pd && minimum; ++axis) {
                  const auto digit = i / strides[axis] % samples[axis];
                  minimum =
                      (!digit || distances[i] < distances[i - strides[axis]])
                      && (digit + 1 == static_cast<std::size_t>(samples[axis])
                          || distances[i] <= distances[i + strides[axis]]);
                }
                if (minimum) {
                  auto pos =
                      std::find_if(starts.begin(), starts.end(), [&](auto j) {
                        return j == sample_count || distances[i] < distances[j];
                      });
                  if (pos != starts.end()) {
                    std::move_backward(pos, starts.end() - 1, starts.end());
                    *pos = i;
                  }
                }
              }
              const auto bounds =
                  extraction.bounds.cspan().subspan(leaf * 2 * pd, 2 * pd);
              for (const auto start : starts) {
                if (start == sample_count) {
                  break;
                }
                double distance, residual;
                solver.SolveFrom(query.data(),
                                 tolerance,
                                 max_iterations,
                                 tight_bounds,
                                 unit_queries.data() + start * pd,
                                 candidate.data(),
                                 point.data(),
                                 difference.data(),
                                 distance,
                                 residual,
                                 nullptr,
                                 nullptr);
                for (std::size_t axis = 0; axis < pd; ++axis) {
                  const double lo = bounds[axis], hi = bounds[pd + axis];
                  const double t = std::clamp(candidate[axis], 0.0, 1.0);
                  mapped[axis] = std::clamp((1.0 - t) * lo + t * hi, lo, hi);
                  // Interior upper faces belong to their right-hand leaf. Keep
                  // a left-hand solution representable even across a
                  // discontinuity.
                  if (mapped[axis] == hi
                      && hi != extraction.finest_breaks[axis].back()) {
                    mapped[axis] = std::nextafter(hi, lo);
                  }
                  const double width = hi - lo;
                  unit[axis] = std::isfinite(width)
                                   ? (mapped[axis] - lo) / width
                                   : (0.5 * mapped[axis] - 0.5 * lo)
                                         / (0.5 * hi - 0.5 * lo);
                }
                Evaluate(view, unit, point, local_options);
                const auto squared = SquaredDistance(point, query);
                if (!std::isnan(squared) && (!found || squared < best)) {
                  found = true;
                  best = squared;
                  best_leaf = leaf;
                  best_residual = residual;
                  std::copy(mapped.begin(), mapped.end(), parameters.begin());
                  std::copy(unit.begin(), unit.end(), best_unit.begin());
                  if (verbose) {
                    std::copy(point.begin(),
                              point.end(),
                              result.points.data() + q * dim);
                  }
                }
              }
              return best;
            },
            arena);

        if (!found) {
          throw SplinepyException(
              "Hierarchy proximity did not find a finite candidate");
        }
        if (!verbose) {
          return;
        }
        result.distances[q] = std::sqrt(best);
        result.residuals[q] = best_residual;
        for (std::size_t c = 0; c < dim; ++c) {
          result.differences[q * dim + c] =
              result.points[q * dim + c] - query[c];
        }
        const auto view = local_view(best_leaf);
        const auto bounds =
            extraction.bounds.cspan().subspan(best_leaf * 2 * pd, 2 * pd);
        auto order = arena.AllocateSpan<int32_t>(pd, 0);
        auto scale = arena.AllocateSpan<double>(pd);
        for (std::size_t axis = 0; axis < pd; ++axis) {
          const double width = bounds[pd + axis] - bounds[axis];
          scale[axis] =
              std::isfinite(width)
                  ? 1.0 / width
                  : 0.5 / (0.5 * bounds[pd + axis] - 0.5 * bounds[axis]);
        }
        for (std::size_t axis = 0; axis < pd; ++axis) {
          ++order[axis];
          auto first = result.first_derivatives.span().subspan(q * first_size
                                                                   + axis * dim,
                                                               dim);
          Derivative(view, best_unit, order, first, local_options);
          for (auto& x : first) {
            if (x != 0.0) {
              x *= scale[axis];
            }
          }
          for (std::size_t other = axis; other < pd; ++other) {
            ++order[other];
            auto second = result.second_derivatives.span().subspan(
                q * second_size + (axis * pd + other) * dim,
                dim);
            Derivative(view, best_unit, order, second, local_options);
            for (auto& x : second) {
              if (x != 0.0) {
                x = (x * scale[axis]) * scale[other];
              }
            }
            if (axis != other) {
              std::copy(second.begin(),
                        second.end(),
                        result.second_derivatives.data() + q * second_size
                            + (other * pd + axis) * dim);
            }
            --order[other];
          }
          --order[axis];
        }
      });
  return result;
}

} // namespace splinepy::core
