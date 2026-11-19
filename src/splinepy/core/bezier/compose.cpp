#include "splinepy/core/bezier/compose.hpp"

#include "splinepy/core/memory/dynamic_buffer.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

#include "splinepy/core/bezier/add.hpp"
#include "splinepy/core/bezier/derivative_spline.hpp"
#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/concurrency/team.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {
namespace {

/// The Bernstein basis of an outer layout composed with an inner spline: row
/// i holds the control points of B_i(inner), outer control point i's basis
/// function, as a scalar Bezier with the given degrees. It depends only on
/// the outer degrees and rationality, never on outer control points, so one
/// basis serves every outer of that layout.
struct CompositionBasis {
  AxisArray<int32_t> degrees; // inner degrees times the outer degree sum
  int64_t size{0};            // control points per row
  Buffer<double> rows;        // one row per outer control point
  Buffer<double> denominator; // weights when only the inner is rational
  bool rational{false};       // outer or inner rational

  const double* Row(int64_t i) const { return rows.data() + i * size; }
};

void ValidateCompose(const SplineView& outer, const SplineView& inner) {
  if (outer.para_dim != inner.dim) {
    throw SplinepyException(
        "Compose: outer.para_dim must equal inner.dim. Got outer.para_dim="
            + std::to_string(outer.para_dim)
            + ", inner.dim=" + std::to_string(inner.dim),
        Status::kDimensionMismatch);
  }
  if (outer.para_dim <= 0 || outer.dim <= 0 || inner.para_dim <= 0
      || inner.dim <= 0) {
    throw SplinepyException("Compose: dimensions must be positive.",
                            Status::kInvalidArgument);
  }
  if (outer.control_points_data == nullptr
      || inner.control_points_data == nullptr) {
    throw SplinepyException("Compose: control points data cannot be null.",
                            Status::kInvalidArgument);
  }
}

/// Builds the basis with the arenas of options' workspace for all
/// intermediates. Only the last, largest stage runs on several threads. Each is
/// a scalar polynomial over the inner domain with degrees k * inner.degrees, so
/// it is kept as (k, coefficients): the homogeneous inner coordinate q_d and w
/// - q_d, their powers, the 1D Bernstein polynomials C(p, i) q^i (w - q)^(p -
/// i) of every outer axis, then one tensor product step per further axis, the
/// last into the rows.
CompositionBasis BuildCompositionBasis(Span<const int32_t> outer_degrees,
                                       bool outer_rational,
                                       const SplineView& inner,
                                       const Options& options) {
  WorkspaceRef ws(options, 64 * 1024, options.nthreads);
  MemoryArena& arena = ws->GetThreadArena(0);
  auto scope = arena.CreateScope();
  const int32_t N = inner.para_dim;
  struct Poly {
    int32_t k;
    double* c;
  };
  const auto degrees_of = [&](int32_t k) {
    AxisArray<int32_t> degrees;
    degrees.resize(N);
    for (int32_t a = 0; a < N; ++a) {
      degrees[a] = k * inner.degrees[a];
    }
    return degrees;
  };
  const auto size_of = [&](int32_t k) {
    int64_t n = 1;
    for (int32_t a = 0; a < N; ++a) {
      n *= k * inner.degrees[a] + 1;
    }
    return n;
  };
  const auto make = [&](int32_t k) {
    return Poly{k, arena.AllocateSpan<double>(size_of(k)).data()};
  };
  const auto multiply =
      [&](Poly x, Poly y, double* out, MemoryArena* scratch = nullptr) {
        detail::MultiplyPolynomialCps(scratch ? *scratch : arena,
                                      N,
                                      degrees_of(x.k),
                                      degrees_of(y.k),
                                      x.c,
                                      1,
                                      y.c,
                                      1,
                                      1,
                                      false,
                                      out);
        return Poly{x.k + y.k, out};
      };
  // base^0 .. base^p, with base's control points value(j)
  const auto powers = [&](auto&& value, int32_t p) {
    Poly* pow = arena.AllocateSpan<Poly>(p + 1).data();
    pow[0] = make(0);
    pow[0].c[0] = 1.0;
    if (p > 0) {
      pow[1] = make(1);
      for (int64_t j = 0; j < inner.num_control_points; ++j) {
        pow[1].c[j] = value(j);
      }
    }
    for (int32_t i = 2; i <= p; ++i) {
      pow[i] = multiply(pow[i - 1], pow[1], make(i).c);
    }
    return pow;
  };
  const auto weight = [&](int64_t j) {
    return inner.IsRational() ? inner.Weight(j) : 1.0;
  };

  const auto M = static_cast<int32_t>(outer_degrees.size());
  int32_t total = 0;
  for (const int32_t p : outer_degrees) {
    total += p;
  }
  CompositionBasis basis;
  basis.rational = outer_rational || inner.IsRational();
  basis.degrees = degrees_of(total);
  basis.size = size_of(total);
  if (inner.IsRational() && !outer_rational) {
    const double* w = powers(weight, total)[total].c;
    basis.denominator.assign(w, w + basis.size);
  }

  Poly* bernstein[kMaxParaDim];
  int64_t n_rows = 1;
  for (int32_t d = 0; d < M; ++d) {
    const int32_t p = outer_degrees[d];
    const auto q = [&](int64_t j) { return inner.Point(j, d) * weight(j); };
    const Poly* Q = powers(q, p);
    const Poly* R = powers([&](int64_t j) { return weight(j) - q(j); }, p);
    Poly* f = bernstein[d] = arena.AllocateSpan<Poly>(p + 1).data();
    f[0] = R[p];
    f[p] = Q[p];
    for (int32_t i = 1; i < p; ++i) {
      f[i] = multiply(Q[i], R[p - i], make(p).c);
      const double binomial = Binomial(p, i);
      for (int64_t j = 0; j < size_of(p); ++j) {
        f[i].c[j] *= binomial;
      }
    }
    n_rows *= p + 1;
  }

  // products over axes 0..d, index i_0 + (p_0 + 1) * (i_1 + ...). The
  // products of a stage share one degree, so their outputs are laid out up
  // front and they run in parallel. The last stage writes the rows.
  basis.rows = Buffer<double>::Uninitialized(n_rows * basis.size);
  const Poly* product = bernstein[0];
  int64_t count = outer_degrees[0] + 1;
  if (M == 1) {
    for (int64_t i = 0; i < count; ++i) {
      std::copy_n(product[i].c, basis.size, basis.rows.data() + i * basis.size);
    }
  }
  for (int32_t d = 1; d < M; ++d) {
    const Poly* f = bernstein[d];
    const int64_t next_count = count * (outer_degrees[d] + 1);
    const int64_t size = size_of(product[0].k + f[0].k);
    double* out = (d == M - 1)
                      ? basis.rows.data()
                      : arena.AllocateSpan<double>(next_count * size).data();
    Poly* next = arena.AllocateSpan<Poly>(next_count).data();
    const double work = static_cast<double>(next_count) * size_of(product[0].k)
                        * size_of(f[0].k);
    ParallelForWithThreadId(int64_t{0},
                            next_count,
                            ThreadsFor(work, options.nthreads),
                            [&](int64_t k, int thread_id) {
                              next[k] =
                                  multiply(product[k % count],
                                           f[k / count],
                                           out + k * size,
                                           &ws->GetThreadArena(thread_id));
                            });
    product = next;
    count = next_count;
  }
  return basis;
}

CompositionBasis ComputeCompositionBasis(const SplineView& outer,
                                         const SplineView& inner,
                                         const Options& options) {
  if (options.validate) {
    ValidateCompose(outer, inner);
    ThrowIfInvalid(inner);
  }
  return BuildCompositionBasis(outer.degrees,
                               outer.IsRational(),
                               inner,
                               options);
}

/// Denominator sum_i w_i B_i of a rational outer spline.
Buffer<double> OuterDenominator(const SplineView& outer,
                                const CompositionBasis& prep) {
  Buffer<double> den(prep.size, 0.0);
  for (int64_t i = 0; i < outer.num_control_points; ++i) {
    const double w = outer.Weight(i);
    const double* b_cps = prep.Row(i);
    for (int64_t j = 0; j < prep.size; ++j) {
      den[j] += w * b_cps[j];
    }
  }
  return den;
}

/// weights of a rational composition
Buffer<double> ResultWeights(const SplineView& outer,
                             const CompositionBasis& prep) {
  return outer.IsRational() ? OuterDenominator(outer, prep) : prep.denominator;
}

/// outer o inner from a ready basis: sum_i P_i B_i, with the weights of a
/// rational result divided out. The basis is only read.
SplineData AccumulateComposition(const SplineView& outer,
                                 const SplineView& inner,
                                 const CompositionBasis& prep) {
  const int32_t N = inner.para_dim;
  const int32_t D = outer.dim;
  const int64_t n_outer = outer.num_control_points;
  const int64_t total_cps = prep.size;

  // channel-major, so that the inner loop runs over contiguous memory
  Buffer<double> acc(total_cps * D, 0.0);
  for (int64_t i = 0; i < n_outer; ++i) {
    const double factor = outer.IsRational() ? outer.Weight(i) : 1.0;
    const double* b_cps = prep.Row(i);
    for (int32_t c = 0; c < D; ++c) {
      const double p_c = outer.Point(i, c) * factor;
      double* a = acc.data() + c * total_cps;
      for (int64_t j = 0; j < total_cps; ++j) {
        a[j] += p_c * b_cps[j];
      }
    }
  }

  SplineData result;
  result.para_dim = N;
  result.dim = D;
  result.degrees.assign(prep.degrees.begin(), prep.degrees.end());
  result.control_points.resize(total_cps * D);

  if (prep.rational) {
    result.weights = ResultWeights(outer, prep);
  }
  for (int64_t j = 0; j < total_cps; ++j) {
    double inv_w = 1.0;
    if (prep.rational) {
      const double w = result.weights[j];
      inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
    }
    for (int32_t c = 0; c < D; ++c) {
      result.control_points[j * D + c] = acc[c * total_cps + j] * inv_w;
    }
  }

  return result;
}

/// d(outer o inner) / d(outer cp i) for every i, from a ready basis.
SplineBatchData SensitivitiesFromBasis(const SplineView& outer,
                                       int32_t para_dim,
                                       const CompositionBasis& prep) {
  const int64_t n_outer = outer.num_control_points;
  const int64_t total_cps = prep.size;

  Buffer<double> denominator;
  if (prep.rational) {
    denominator = ResultWeights(outer, prep);
  }

  SplineBatchData batch(n_outer);
  for (int64_t i = 0; i < n_outer; ++i) {
    SplineData& sens = batch[i];
    sens.para_dim = para_dim;
    sens.dim = 1;
    sens.degrees.assign(prep.degrees.begin(), prep.degrees.end());

    const double* b_cps = prep.Row(i);
    if (prep.rational) {
      sens.weights = denominator;
      sens.control_points.resize(total_cps);
      const double factor = outer.IsRational() ? outer.Weight(i) : 1.0;
      for (int64_t j = 0; j < total_cps; ++j) {
        const double w = sens.weights[j];
        const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
        sens.control_points[j] = (factor * b_cps[j]) * inv_w;
      }
    } else {
      sens.control_points.assign(b_cps, b_cps + total_cps);
    }
  }
  return batch;
}

/// Outer layout a batch needs a basis for: degrees and rationality.
struct OuterLayout {
  AxisArray<int32_t> degrees; // by value: layouts outlive the views they come from
  bool rational;
};

/// Multiply-adds of a batch: basis products, then one weighted sum per pair.
double BatchWork(Span<const OuterLayout> layouts,
                 SplineBatchView inners,
                 std::size_t n_outer) {
  double work = 0.0;
  for (std::size_t i = 0; i < inners.size(); ++i) {
    for (const auto& layout : layouts) {
      double outer_cps = 1.0;
      double composed_cps = 1.0;
      for (std::size_t d = 0; d < layout.degrees.size(); ++d) {
        const double p = layout.degrees[d];
        outer_cps *= p + 1.0;
        const double q =
            d < inners[i].degrees.size() ? inners[i].degrees[d] : 1.0;
        composed_cps *= p * q + 1.0;
      }
      work += outer_cps * composed_cps
              * (composed_cps + 3.0 * static_cast<double>(n_outer));
    }
  }
  return work;
}

/// Runs pair(o, i, preps, worker) for every (outer, inner) pair of a batch,
/// where preps are the bases of inner i for each of the layouts and worker
/// are single threaded options with the thread's own workspace and validation
/// off. A call starts at most ResolveThreadCount(options.nthreads) threads,
/// the caller included, fewer for small batches (see ThreadsFor), and
/// never nests parallel loops.
///
/// With at least as many inners as threads each thread owns whole inners: it
/// builds their bases alone and runs all their outers before the next, so
/// memory stays at nthreads inners' bases. Otherwise the bases are built one
/// after another, each stage on all threads, and then the team shares the
/// n_inner * n_outer pairs.
template<typename Pair>
void ForEachPair(Span<const OuterLayout> layouts,
                 SplineBatchView inners,
                 std::size_t n_outer,
                 const Options& options,
                 Pair&& pair) {
  const std::size_t n_inner = inners.size();
  const std::size_t n_layouts = layouts.size();
  const int nthreads =
      ThreadsFor(BatchWork(layouts, inners, n_outer), options.nthreads);
  Options local = options;
  local.validate = false;
  local.nthreads = nthreads;
  WorkspaceRef ws(local, 16 * 1024, nthreads);

  const auto build = [&](std::size_t i, std::size_t l, const Options& worker) {
    return BuildCompositionBasis(layouts[l].degrees,
                                 layouts[l].rational,
                                 inners[i],
                                 worker);
  };

  if (n_inner >= static_cast<std::size_t>(nthreads)) {
    ParallelForWithThreadId(
        std::size_t{0}, n_inner, nthreads, [&](std::size_t i, int thread_id) {
          const Options worker = WorkerOptions(local, *ws, thread_id);
          Buffer<CompositionBasis> bases(n_layouts);
          for (std::size_t l = 0; l < n_layouts; ++l) {
            bases[l] = build(i, l, worker);
          }
          for (std::size_t o = 0; o < n_outer; ++o) {
            pair(o, i, bases.data(), worker);
          }
        });
    return;
  }

  // fewer inners than threads: each basis uses the threads itself
  Buffer<CompositionBasis> bases(n_inner * n_layouts);
  for (std::size_t k = 0; k < bases.size(); ++k) {
    bases[k] = build(k / n_layouts, k % n_layouts, local);
  }
  RunTeam(nthreads, [&](Team& team) {
    const Options worker = WorkerOptions(local, *ws, team.Id());
    // pairs grouped by inner: k = i * n_outer + o
    team.For(n_inner * n_outer, [&](std::size_t k) {
      const std::size_t i = k / n_outer;
      pair(k % n_outer, i, &bases[i * n_layouts], worker);
    });
  });
}

/// The batch functions share their basis between outers of identical layout,
/// e.g. the Bezier elements of one refined spline.
void CheckSharedLayout(SplineBatchView outers,
                       SplineBatchView inners,
                       bool validate) {
  const SplineView& ref = outers[0];
  for (std::size_t o = 0; o < outers.size(); ++o) {
    const SplineView& outer = outers[o];
    bool same = outer.para_dim == ref.para_dim && outer.dim == ref.dim
                && outer.IsRational() == ref.IsRational();
    for (int32_t d = 0; same && d < ref.para_dim; ++d) {
      same = outer.degrees[d] == ref.degrees[d];
    }
    if (!same) {
      throw SplinepyException(
          "Compose batch: all outer splines must share para_dim, dim, degrees "
          "and rationality.",
          Status::kDimensionMismatch);
    }
  }
  if (validate) {
    for (std::size_t i = 0; i < inners.size(); ++i) {
      ThrowIfInvalid(inners[i]);
    }
    for (std::size_t o = 0; o < outers.size(); ++o) {
      for (std::size_t i = 0; i < inners.size(); ++i) {
        ValidateCompose(outers[o], inners[i]);
      }
    }
  }
}

void ValidatePairOffsets(std::size_t outer_count,
                         std::size_t tile_count,
                         Span<const std::size_t> offsets) {
  if (offsets.size() != outer_count + 1 || offsets.front() != 0
      || offsets.back() != tile_count
      || !std::is_sorted(offsets.begin(), offsets.end())) {
    throw std::invalid_argument("Invalid offsets for paired composition.");
  }
}

bool SameTileGeometry(const SplineData& a, const SplineData& b) {
  return a.para_dim == b.para_dim && a.dim == b.dim
         && std::equal(a.degrees.begin(),
                       a.degrees.end(),
                       b.degrees.begin(),
                       b.degrees.end())
         && std::equal(a.control_points.begin(),
                       a.control_points.end(),
                       b.control_points.begin(),
                       b.control_points.end())
         && std::equal(a.weights.begin(),
                       a.weights.end(),
                       b.weights.begin(),
                       b.weights.end());
}

/// sparse row: (coarse control point, coefficient)
using ExtractionRow = Buffer<std::pair<int64_t, double>>;

void AssembleMacroFields(Span<const ExtractionRow> rows,
                         const SplineData& geometry,
                         const CompositionBasis& basis,
                         std::size_t output,
                         Buffer<SplineBatchData>& fields) {
  const auto count = basis.size;
  std::map<int64_t, Buffer<double>> mapped;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const double* values = basis.Row(local);
    for (const auto& entry : rows[local]) {
      auto& destination =
          mapped.try_emplace(entry.first, count, 0.0).first->second;
      for (int64_t j = 0; j < count; ++j) {
        destination[j] += entry.second * values[j];
      }
    }
  }

  for (const auto& [coarse, values] : mapped) {
    if (std::all_of(values.begin(), values.end(), [](double value) {
          return value == 0.0;
        })) {
      continue;
    }
    for (int32_t c = 0; c < geometry.dim; ++c) {
      auto& field = fields[coarse * geometry.dim + c][output];
      field.para_dim = geometry.para_dim;
      field.dim = geometry.dim;
      field.degrees.assign(basis.degrees.begin(), basis.degrees.end());
      field.weights = geometry.weights;
      field.control_points = Buffer<double>(count * geometry.dim, 0.0);
      for (int64_t j = 0; j < count; ++j) {
        const double value = values[j];
        if (basis.rational) {
          const double weight = field.weights[j];
          field.control_points[j * geometry.dim + c] =
              weight == 0.0 ? 0.0 : value / weight;
        } else {
          field.control_points[j * geometry.dim + c] = value;
        }
      }
    }
  }
}

} // namespace

struct MicrostructurePlan::Data {
  struct TileBasis {
    SplineData tile;
    CompositionBasis basis;
  };

  struct PairedCache {
    Buffer<std::size_t> offsets;
    Buffer<std::shared_ptr<const TileBasis>> entries;
    std::size_t reused{0};
  };

  mutable std::shared_ptr<const PairedCache> paired_cache;
  Buffer<int32_t> degrees;
  Buffer<int32_t> tile_dimensions;
  Buffer<CompositionBasis> bases;
  Buffer<Buffer<ExtractionRow>> extraction;
  int64_t control_count;
  int32_t dim;
};

MicrostructurePlan::MicrostructurePlan(
    const SplineView& outer,
    SplineBatchView tiles,
    Span<const Span<const double>> additional_knots,
    const Options& options) {
  ThrowIfInvalid(outer);
  if (!outer.HasKnotVectors() || outer.IsRational()
      || additional_knots.size() != outer.degrees.size()) {
    throw std::invalid_argument("Expected a polynomial B-spline and "
                                "additional knots per axis.");
  }

  auto data = std::make_shared<Data>();
  data->degrees.assign(outer.degrees.begin(), outer.degrees.end());
  data->control_count = outer.num_control_points;
  data->dim = outer.dim;
  data->extraction = Buffer<Buffer<ExtractionRow>>(1);
  data->extraction[0] = Buffer<ExtractionRow>(1);
  data->extraction[0][0] = ExtractionRow{{0, 1.0}};
  int64_t stride = 1;

  for (int axis = 0; axis < outer.para_dim; ++axis) {
    const auto knots = outer.knot_vectors[axis];
    const auto degree = outer.degrees[axis];
    if (knots[degree] != knots.front()
        || knots[knots.size() - degree - 1] != knots.back()) {
      throw std::invalid_argument("Expected clamped knots.");
    }

    // Extract an identity control net to recover the one-dimensional map.
    const auto count = outer.NumControlPointsPerAxis(axis);
    if (count > INT32_MAX) {
      throw std::invalid_argument("Too many control points on one axis.");
    }
    SplineData identity;
    identity.para_dim = 1;
    identity.dim = static_cast<int32_t>(count);
    identity.degrees = Buffer<int32_t>(1, degree);
    identity.knot_vectors.resize(1);
    identity.knot_vectors[0].assign(knots.begin(), knots.end());
    identity.control_points = Buffer<double>(count * count, 0.);
    for (int64_t i = 0; i < count; ++i) {
      identity.control_points[i * count + i] = 1.;
    }
    auto refined =
        InsertKnots(identity.View(), 0, additional_knots[axis], options);
    auto axis_patches = ExtractBezierPatches(refined.View(), options);
    Buffer<Buffer<ExtractionRow>> next(axis_patches.size()
                                       * data->extraction.size());
    std::size_t element = 0;
    for (const auto& patch : axis_patches) {
      const auto view = patch.View();
      for (const auto& previous : data->extraction) {
        auto& rows = next[element++];
        rows = Buffer<ExtractionRow>((static_cast<std::size_t>(degree) + 1)
                                     * previous.size());
        DynamicBuffer<std::pair<int64_t, double>> combined;
        for (int32_t j = 0; j <= degree; ++j) {
          for (std::size_t i = 0; i < previous.size(); ++i) {
            for (int64_t k = 0; k < count; ++k) {
              const double coefficient = view.Point(j, k);
              if (coefficient != 0.) {
                for (const auto& entry : previous[i]) {
                  combined.emplace_back(entry.first + stride * k,
                                        entry.second * coefficient);
                }
              }
            }
            rows[j * previous.size() + i] = std::move(combined).Finish();
          }
        }
      }
    }
    stride *= count;
    data->extraction = std::move(next);
  }

  data->tile_dimensions.resize(tiles.size());
  data->bases.resize(tiles.size());
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    const auto& tile = tiles[i];
    if (tile.HasKnotVectors()) {
      throw std::invalid_argument("Expected Bézier tile patches.");
    }
    data->tile_dimensions[i] = tile.para_dim;
    data->bases[i] = ComputeCompositionBasis(outer, tile, options);
  }
  data_ = std::move(data);
}

int64_t MicrostructurePlan::NumControlPoints() const {
  return data_->control_count;
}

int32_t MicrostructurePlan::Dim() const { return data_->dim; }

SplineBatchData MicrostructurePlan::Apply(Span<const double> control_points,
                                          const Options& options) const {
  return ApplyFixed(control_points, nullptr, options);
}

SplineBatchData MicrostructurePlan::ApplyWithMacroSensitivities(
    Span<const double> control_points,
    Buffer<SplineBatchData>& fields,
    const Options& options) const {
  if (data_->bases.empty()) {
    throw std::invalid_argument("Macro sensitivities require fixed tiles.");
  }
  return ApplyFixed(control_points, &fields, options);
}

SplineBatchData
MicrostructurePlan::ApplyFixed(Span<const double> control_points,
                               Buffer<SplineBatchData>* fields,
                               const Options& options) const {
  const auto& data = *data_;
  if (control_points.size()
          != static_cast<std::size_t>(data.control_count * data.dim)
      || !std::all_of(control_points.begin(),
                      control_points.end(),
                      [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        "Expected finite macro control points with the prepared shape.");
  }

  const auto tile_count = data.bases.size();
  SplineBatchData result(data.extraction.size()
                         * std::max(tile_count, std::size_t{1}));
  if (fields) {
    fields->resize(data.control_count * data.dim);
    for (auto& field : *fields) {
      field.resize(result.size());
    }
  }
  ParallelFor(
      int64_t{0},
      static_cast<int64_t>(data.extraction.size()),
      options.nthreads,
      [&](int64_t element) {
        const auto& rows = data.extraction[element];
        Buffer<double> points(rows.size() * data.dim, 0.);
        for (std::size_t i = 0; i < rows.size(); ++i) {
          for (const auto& entry : rows[i]) {
            for (int32_t c = 0; c < data.dim; ++c) {
              points[i * data.dim + c] +=
                  entry.second * control_points[entry.first * data.dim + c];
            }
          }
        }
        SplineView outer;
        outer.para_dim = static_cast<int32_t>(data.degrees.size());
        outer.dim = data.dim;
        outer.degrees = data.degrees;
        outer.num_control_points = static_cast<int64_t>(rows.size());
        outer.control_points_data = points.data();
        outer.cp_stride = data.dim;
        if (tile_count == 0) {
          auto& patch = result[element];
          patch.para_dim = outer.para_dim;
          patch.dim = outer.dim;
          patch.degrees = data.degrees;
          patch.control_points = std::move(points);
          return;
        }
        for (std::size_t tile = 0; tile < tile_count; ++tile) {
          SplineView inner;
          inner.para_dim = data.tile_dimensions[tile];
          const auto output = element * tile_count + tile;
          const auto& basis = data.bases[tile];
          result[output] = AccumulateComposition(outer, inner, basis);
          if (!fields) {
            continue;
          }

          AssembleMacroFields(rows, result[output], basis, output, *fields);
        }
      });
  return result;
}

SplineBatchData
MicrostructurePlan::ApplyPaired(SplineBatchView outers,
                                Span<const SplineData> tiles,
                                Span<const std::size_t> offsets,
                                const Options& options,
                                Buffer<SplineBatchData>* macro_fields) const {
  ValidatePairOffsets(outers.size(), tiles.size(), offsets);
  if (outers.size() != data_->extraction.size() || !data_->bases.empty()) {
    throw std::invalid_argument("Expected the prepared set of macro patches.");
  }

  auto previous = std::atomic_load(&data_->paired_cache);
  auto next = std::make_shared<Data::PairedCache>();
  next->offsets.assign(offsets.begin(), offsets.end());
  next->entries.resize(tiles.size());

  // Bound retained numerical coefficients. Uncached entries are composed
  // normally.
  double remaining = 64. * 1024 * 1024;
  const auto degree_sum =
      std::accumulate(data_->degrees.begin(), data_->degrees.end(), 0.0);
  Buffer<uint8_t> retain(tiles.size());
  for (std::size_t i = 0; i < outers.size(); ++i) {
    const auto& outer = outers[i];
    if (outer.IsRational() || outer.HasKnotVectors() || outer.dim != data_->dim
        || !std::equal(outer.degrees.begin(),
                       outer.degrees.end(),
                       data_->degrees.begin(),
                       data_->degrees.end())) {
      throw std::invalid_argument("Expected the prepared macro patch layout.");
    }
    for (std::size_t j = offsets[i]; j < offsets[i + 1]; ++j) {
      const auto& tile = tiles[j];
      double bytes = sizeof(double) * (outer.num_control_points + 1.0);
      for (const auto degree : tile.degrees) {
        bytes *= 1.0 + degree * degree_sum;
      }
      if (!tile.HasKnotVectors() && bytes <= remaining) {
        retain[j] = 1;
        remaining -= bytes;
      }
    }
  }

  Options cell_options = options;
  cell_options.nthreads = 1;
  cell_options.workspace = nullptr;
  SplineBatchData results(tiles.size());
  if (macro_fields) {
    macro_fields->resize(data_->control_count * data_->dim);
    for (auto& field : *macro_fields) {
      field.resize(results.size());
    }
  }
  Buffer<std::size_t> hits(outers.size());
  ParallelFor(
      std::size_t{0},
      outers.size(),
      options.nthreads,
      [&](std::size_t i) {
        const auto& outer = outers[i];
        for (std::size_t j = offsets[i]; j < offsets[i + 1]; ++j) {
          const auto inner = tiles[j].View();
          if (!retain[j]) {
            auto basis = ComputeCompositionBasis(outer, inner, cell_options);
            results[j] = AccumulateComposition(outer, inner, basis);
            if (macro_fields) {
              AssembleMacroFields(data_->extraction[i],
                                  results[j],
                                  basis,
                                  j,
                                  *macro_fields);
            }
            continue;
          }
          auto& entry = next->entries[j];
          if (previous) {
            const auto slot = j - offsets[i];
            const auto old_start = previous->offsets[i];
            if (slot < previous->offsets[i + 1] - old_start) {
              const auto& old = previous->entries[old_start + slot];
              if (old && SameTileGeometry(old->tile, tiles[j])) {
                entry = old;
                ++hits[i];
              }
            }
          }
          if (!entry) {
            entry = std::make_shared<const Data::TileBasis>(Data::TileBasis{
                tiles[j],
                ComputeCompositionBasis(outer, inner, cell_options)});
          }
          results[j] = AccumulateComposition(outer, inner, entry->basis);
          if (macro_fields) {
            AssembleMacroFields(data_->extraction[i],
                                results[j],
                                entry->basis,
                                j,
                                *macro_fields);
          }
        }
      });
  next->reused = std::accumulate(hits.begin(), hits.end(), std::size_t{0});
  std::shared_ptr<const Data::PairedCache> published = std::move(next);
  std::atomic_store(&data_->paired_cache, std::move(published));
  return results;
}

std::size_t MicrostructurePlan::NumReusedBases() const {
  const auto cache = std::atomic_load(&data_->paired_cache);
  return cache ? cache->reused : 0;
}

SplineData Compose(const SplineView& outer,
                   const SplineView& inner,
                   const Options& options) {
  const auto prep = ComputeCompositionBasis(outer, inner, options);
  return AccumulateComposition(outer, inner, prep);
}

SplineBatchData ComposePaired(SplineBatchView outers,
                              SplineBatchView tiles,
                              Span<const std::size_t> offsets,
                              const Options& options) {
  ValidatePairOffsets(outers.size(), tiles.size(), offsets);
  Options cell_options = options;
  cell_options.nthreads = 1;
  cell_options.workspace = nullptr;
  SplineBatchData results(tiles.size());
  ParallelFor(std::size_t{0},
              outers.size(),
              options.nthreads,
              [&](std::size_t i) {
                for (std::size_t j = offsets[i]; j < offsets[i + 1]; ++j) {
                  results[j] = Compose(outers[i], tiles[j], cell_options);
                }
              });
  return results;
}

Buffer<SplineBatchData>
ComposeParameterFields(SplineBatchView outers,
                       SplineBatchView tiles,
                       Span<const std::size_t> offsets,
                       Span<const SplineData> geometry,
                       Span<const TileParameterDerivative> derivatives,
                       std::size_t parameter_count,
                       const Options& options) {
  ValidatePairOffsets(outers.size(), tiles.size(), offsets);
  if (geometry.size() != tiles.size()) {
    throw std::invalid_argument("Expected geometry for every tile patch.");
  }
  std::pair<std::size_t, std::size_t> previous{};
  bool first = true;
  for (const auto& group : derivatives) {
    const auto key = std::make_pair(group.cell, group.parameter);
    if (group.cell >= outers.size() || group.parameter >= parameter_count
        || (!first && key <= previous)
        || group.patches.size()
               != offsets[group.cell + 1] - offsets[group.cell]) {
      throw std::invalid_argument("Invalid parameter sensitivity group.");
    }
    previous = key;
    first = false;
  }

  Buffer<SplineBatchData> fields(parameter_count);
  for (auto& field : fields) {
    field.resize(geometry.size());
  }
  Options local = options;
  local.nthreads = 1;
  local.workspace = nullptr;
  ParallelFor(std::size_t{0},
              derivatives.size(),
              options.nthreads,
              [&](std::size_t i) {
                const auto& group = derivatives[i];
                if (group.patches.empty()) {
                  return;
                }
                const auto offset = offsets[group.cell];
                auto patches = CompositionDerivativeBatch(
                    {outers.data() + group.cell, 1},
                    {tiles.data() + offset, group.patches.size()},
                    group.patches,
                    local);
                for (std::size_t j = 0; j < patches.size(); ++j) {
                  fields[group.parameter][offset + j] = std::move(patches[j]);
                }
              });

  return fields;
}

SplineBatchData ComposeBatch(SplineBatchView outers,
                             SplineBatchView inners,
                             const Options& options) {
  const std::size_t n_outer = outers.size();
  const std::size_t n_inner = inners.size();
  SplineBatchData result(n_outer * n_inner);
  if (n_outer == 0 || n_inner == 0) {
    return result;
  }
  CheckSharedLayout(outers, inners, options.validate);

  // result[o * n_inner + i] = outers[o] o inners[i]
  const OuterLayout layout{outers[0].degrees, outers[0].IsRational()};
  ForEachPair(Span<const OuterLayout>(&layout, 1),
              inners,
              n_outer,
              options,
              [&](std::size_t o,
                  std::size_t i,
                  const CompositionBasis* preps,
                  const Options&) {
                result[o * n_inner + i] =
                    AccumulateComposition(outers[o], inners[i], preps[0]);
              });
  return result;
}

Buffer<SplineBatchData> ComposeSensitivitiesBatch(SplineBatchView outers,
                                                  SplineBatchView inners,
                                                  const Options& options) {
  const std::size_t n_outer = outers.size();
  const std::size_t n_inner = inners.size();
  Buffer<SplineBatchData> result(n_outer * n_inner);
  if (n_outer == 0 || n_inner == 0) {
    return result;
  }
  CheckSharedLayout(outers, inners, options.validate);

  // result[o * n_inner + i] = ComposeSensitivities(outers[o], inners[i])
  const OuterLayout layout{outers[0].degrees, outers[0].IsRational()};
  ForEachPair(
      Span<const OuterLayout>(&layout, 1),
      inners,
      n_outer,
      options,
      [&](std::size_t o,
          std::size_t i,
          const CompositionBasis* preps,
          const Options&) {
        result[o * n_inner + i] =
            SensitivitiesFromBasis(outers[o], inners[i].para_dim, preps[0]);
      });
  return result;
}

SplineBatchData CompositionDerivativeBatch(SplineBatchView outers,
                                           SplineBatchView inners,
                                           SplineBatchView inner_derivatives,
                                           const Options& options) {
  const std::size_t n_outer = outers.size();
  const std::size_t n_inner = inners.size();
  if (inner_derivatives.size() != n_inner) {
    throw SplinepyException("CompositionDerivativeBatch: one inner "
                            "derivative per inner is needed.",
                            Status::kInvalidArgument);
  }
  SplineBatchData result(n_outer * n_inner);
  if (n_outer == 0 || n_inner == 0) {
    return result;
  }
  CheckSharedLayout(outers, inners, options.validate);
  if (options.validate) {
    for (std::size_t i = 0; i < n_inner; ++i) {
      if (inners[i].dim != inner_derivatives[i].dim
          || inners[i].para_dim != inner_derivatives[i].para_dim) {
        throw SplinepyException("CompositionDerivativeBatch: inner "
                                "derivative must have the dim and "
                                "para_dim of its inner.",
                                Status::kDimensionMismatch);
      }
    }
  }

  const int32_t M = outers[0].para_dim;
  const auto m = static_cast<std::size_t>(M);
  // orders[d * M + e]: first derivative along d
  Buffer<int32_t> orders(m * m, 0);
  for (int32_t d = 0; d < M; ++d) {
    orders[d * M + d] = 1;
  }
  auto derivative_of =
      [&](const SplineView& outer, int32_t d, const Options& kernel_options) {
        return DerivativeSpline(
            outer, Span<const int32_t>(&orders[d * M], m), kernel_options);
      };

  // The derivative splines of all outers share their degrees, so one basis per
  // (inner, direction) serves every outer.
  Buffer<SplineData> layout_der(m);
  Buffer<OuterLayout> layouts(m);
  for (int32_t d = 0; d < M; ++d) {
    layout_der[d] = derivative_of(outers[0], d, options);
    const SplineView der_view = layout_der[d].View();
    layouts[d] = {der_view.degrees, der_view.IsRational()};
  }

  // inner_der_d[i * M + d]: d-th component of inner derivative i, cheap copies
  Options serial = options;
  serial.validate = false;
  serial.nthreads = 1;
  Buffer<SplineData> inner_der_d(n_inner * m);
  for (std::size_t i = 0; i < n_inner; ++i) {
    for (int32_t d = 0; d < M; ++d) {
      inner_der_d[i * m + d] = ExtractDim(inner_derivatives[i], d, serial);
    }
  }

  // result[o * n_inner + i] = CompositionDerivative(outers[o], inners[i],
  // inner_derivatives[i])
  ForEachPair(Span<const OuterLayout>(layouts.data(), m),
              inners,
              n_outer,
              options,
              [&](std::size_t o,
                  std::size_t i,
                  const CompositionBasis* preps,
                  const Options& worker) {
                // the outers are validated as given by the caller
                Options checked = worker;
                checked.validate = options.validate;
                SplineData sum;
                for (int32_t d = 0; d < M; ++d) {
                  SplineData outer_der = derivative_of(outers[o], d, checked);
                  SplineData comp = AccumulateComposition(outer_der.View(),
                                                          inners[i],
                                                          preps[d]);
                  SplineData term =
                      Multiply(comp.View(), inner_der_d[i * m + d].View(), worker);
                  sum = (d == 0) ? std::move(term)
                                 : Add(sum.View(), term.View(), worker);
                }
                result[o * n_inner + i] = std::move(sum);
              });
  return result;
}

SplineBatchData ComposeSensitivities(const SplineView& outer,
                                     const SplineView& inner,
                                     const Options& options) {
  auto prep = ComputeCompositionBasis(outer, inner, options);
  return SensitivitiesFromBasis(outer, inner.para_dim, prep);
}

SplineData CompositionDerivative(const SplineView& outer,
                                 const SplineView& inner,
                                 const SplineView& inner_derivative,
                                 const Options& options) {
  if (options.validate) {
    if (outer.para_dim != inner.dim) {
      throw SplinepyException(
          "CompositionDerivative: outer.para_dim must equal inner.dim.",
          Status::kDimensionMismatch);
    }
    if (inner.dim != inner_derivative.dim) {
      throw SplinepyException(
          "CompositionDerivative: inner.dim must equal inner_derivative.dim.",
          Status::kDimensionMismatch);
    }
    if (inner.para_dim != inner_derivative.para_dim) {
      throw SplinepyException(
          "CompositionDerivative: inner.para_dim must equal "
          "inner_derivative.para_dim.",
          Status::kDimensionMismatch);
    }
  }

  const int32_t M = outer.para_dim;
  AxisArray<int32_t> orders;
  orders.resize(M);
  SplineData result;

  for (int32_t d = 0; d < M; ++d) {
    orders[d] = 1;
    SplineData outer_der =
        DerivativeSpline(outer,
                         Span<const int32_t>(orders.data(), orders.size()),
                         options);
    orders[d] = 0;

    SplineData comp_d = Compose(outer_der.View(), inner, options);
    SplineData inner_der_d = ExtractDim(inner_derivative, d, options);
    SplineData term = Multiply(comp_d.View(), inner_der_d.View(), options);

    result =
        (d == 0) ? std::move(term) : Add(result.View(), term.View(), options);
  }

  return result;
}

} // namespace splinepy::core
