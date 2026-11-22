/*
MIT License

Copyright (c) 2021 Jaewook Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Search state, cost function J = .5 |spline(u) - query|^2 and its
// derivatives, shared by all proximity solvers.

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <optional>

#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/proximity/detail/search.hpp"

namespace splinepy::proximity {
namespace detail {

MemoryArena& QueryArena() {
  return splinepy::core::DefaultWorkspace().GetThreadArena(0);
}

const splinepy::core::Options& QueryOptions() {
  // no workspace: kernels take the calling thread's default one, the same
  // QueryArena allocates from; both release what they took when they return
  static const splinepy::core::Options options = []() {
    splinepy::core::Options o;
    o.validate = false;
    return o;
  }();
  return options;
}

SearchData::SearchData(MemoryArena& arena_,
                       const int para_dim_,
                       const int dim_,
                       const double tolerance,
                       const int max_iter,
                       const double* query,
                       double* final_guess,
                       double* nearest,
                       double* nearest_minus_query,
                       double* first_derivatives,
                       double* second_derivatives)
    : para_dim(para_dim_),
      dim(dim_),
      arena(&arena_) {
  options.tolerance = tolerance;
  options.max_iter = (max_iter < 0) ? para_dim * 20 : max_iter;
  options.convergence_goal = 0.0;

  status.J = -1.0;
  status.distance = -1.0;
  status.squared_distance = -1.0;
  status.convergence_norm = -1.0;
  status.settled = false;

  const auto output = [&](double* pointer, std::size_t size) {
    return pointer ? Span<double>(pointer, size)
                   : arena->AllocateSpan<double>(size);
  };
  phys_query = Span<const double>(query, dim);
  current_guess = output(final_guess, para_dim);
  current_phys = output(nearest, dim);
  difference = output(nearest_minus_query, dim);
  spline_gradient = output(first_derivatives, para_dim * dim);
  spline_hessian = output(second_derivatives, para_dim * para_dim * dim);

  initial_guess = arena->AllocateSpan<double>(para_dim);
  d2jdu2 = arena->AllocateSpan<double>(para_dim * para_dim);
  djdu = arena->AllocateSpan<double>(para_dim);
  delta_guess = arena->AllocateSpan<double>(para_dim);
  spline_gradient_AAt = arena->AllocateSpan<double>(para_dim * para_dim);
  search_bounds = arena->AllocateSpan<double>(2 * para_dim);
  lower_bound = search_bounds.subspan(0, para_dim);
  upper_bound = search_bounds.subspan(para_dim, para_dim);
  clipped = arena->AllocateSpan<int>(para_dim);
  computed_at = arena->AllocateSpan<double>(para_dim);
}

bool Cholesky(Span<double> a, const int n, const double min_pivot) {
  for (int j{}; j < n; ++j) {
    double d = a[j * n + j];
    for (int k{}; k < j; ++k) {
      d -= a[j * n + k] * a[j * n + k];
    }
    if (!(d > min_pivot)) {
      return false;
    }
    const double l_jj = std::sqrt(d);
    a[j * n + j] = l_jj;
    for (int i{j + 1}; i < n; ++i) {
      double v = a[i * n + j];
      for (int k{}; k < j; ++k) {
        v -= a[i * n + k] * a[j * n + k];
      }
      a[i * n + j] = v / l_jj;
    }
  }
  return true;
}

void CholeskySolve(Span<const double> l,
                   const int n,
                   Span<const double> b,
                   Span<double> x) {
  std::copy_n(b.begin(), n, x.begin());
  for (int i{}; i < n; ++i) {
    for (int k{}; k < i; ++k) {
      x[i] -= l[i * n + k] * x[k];
    }
    x[i] /= l[i * n + i];
  }
  for (int i{n - 1}; i >= 0; --i) {
    for (int k{i + 1}; k < n; ++k) {
      x[i] -= l[k * n + i] * x[k];
    }
    x[i] /= l[i * n + i];
  }
}

bool SolvePivoted(Span<double> matrix,
                  Span<double> rhs,
                  Span<double> solution,
                  Span<int> rows) {
  const int n = static_cast<int>(rows.size());
  std::iota(rows.begin(), rows.end(), 0);
  for (int i{}; i < n; ++i) {
    int pivot = i;
    for (int j{i + 1}; j < n; ++j) {
      if (std::abs(matrix[rows[j] * n + i])
          > std::abs(matrix[rows[pivot] * n + i])) {
        pivot = j;
      }
    }
    std::swap(rows[i], rows[pivot]);
    const auto pivot_row = matrix.subspan(rows[i] * n, n);
    if (pivot_row[i] == 0.0 || !std::isfinite(pivot_row[i])) {
      return false;
    }
    for (int j{i + 1}; j < n; ++j) {
      const auto row = matrix.subspan(rows[j] * n, n);
      const double factor = row[i] / pivot_row[i];
      row[i] = 0.0;
      for (int k{i + 1}; k < n; ++k) {
        row[k] -= pivot_row[k] * factor;
      }
      rhs[rows[j]] -= rhs[rows[i]] * factor;
    }
  }

  for (int i{n - 1}; i >= 0; --i) {
    const auto row = matrix.subspan(rows[i] * n, n);
    double sum = 0.0;
    for (int j{i + 1}; j < n; ++j) {
      sum += row[j] * solution[j];
    }
    solution[i] = (rhs[rows[i]] - sum) / row[i];
    if (!std::isfinite(solution[i])) {
      return false;
    }
  }
  return true;
}

void ClipGuess(SearchData& data) {
  for (int axis{}; axis < data.para_dim; ++axis) {
    double& value = data.current_guess[axis];
    const double lower = data.lower_bound[axis];
    const double upper = data.upper_bound[axis];
    data.clipped[axis] = value > upper ? 1 : value < lower ? -1 : 0;
    if (data.clipped[axis] != 0) {
      value = data.clipped[axis] > 0 ? upper : lower;
    }
  }
}

void ComputeGramMatrix(SearchData& data) {
  const int n = data.para_dim;
  for (int i{}; i < n; ++i) {
    const auto row_i = data.spline_gradient.subspan(i * data.dim, data.dim);
    for (int j{i}; j < n; ++j) {
      const auto row_j = data.spline_gradient.subspan(j * data.dim, data.dim);
      const double value = Dot(row_i, row_j);
      data.spline_gradient_AAt[i * n + j] = value;
      data.spline_gradient_AAt[j * n + i] = value;
    }
  }
}

namespace {

/// derivative orders 0, e_i and, if depth > 1, e_i + e_j for i <= j
splinepy::core::Buffer<int32_t> SearchOrders(const int para_dim,
                                             const int depth) {
  const int n_orders =
      1 + para_dim + (depth > 1 ? para_dim * (para_dim + 1) / 2 : 0);
  splinepy::core::Buffer<int32_t> orders(n_orders * para_dim);
  int32_t* order = orders.data() + para_dim;
  for (int i{}; i < para_dim; ++i, order += para_dim) {
    order[i] = 1;
  }
  if (depth > 1) {
    for (int i{}; i < para_dim; ++i) {
      for (int j{i}; j < para_dim; ++j, order += para_dim) {
        ++order[i];
        ++order[j];
      }
    }
  }
  return orders;
}

/// schedule of SearchOrders (depth 1 or 2), built once per thread and kind of
/// spline
const splinepy::core::DerivativeOrders&
SearchSchedule(const int para_dim, const bool rational, const int depth) {
  thread_local std::optional<splinepy::core::DerivativeOrders>
      cache[splinepy::core::kMaxParaDim][2][2];
  auto& entry = cache[para_dim - 1][rational][depth - 1];
  if (!entry) {
    entry.emplace(SearchOrders(para_dim, depth).cspan(),
                  static_cast<std::size_t>(para_dim),
                  rational);
  }
  return *entry;
}

} // namespace

void SplineDerivatives(const splinepy::core::SplineView& spline,
                       Span<const double> at,
                       const int depth,
                       double* value,
                       double* first,
                       double* second) {
  const int para_dim = spline.para_dim;
  const int dim = spline.dim;
  if (depth <= 0) {
    splinepy::core::Evaluate(spline,
                             Span<const double>(at.data(), para_dim),
                             Span<double>(value, dim),
                             QueryOptions());
    return;
  }
  const int d = depth > 1 ? 2 : 1;
  const int n_orders =
      1 + para_dim + (d > 1 ? para_dim * (para_dim + 1) / 2 : 0);
  auto scope = QueryArena().CreateScope();
  auto out = QueryArena().AllocateSpan<double>(n_orders * dim);
  splinepy::core::Derivative(spline,
                             Span<const double>(at.data(), para_dim),
                             SearchSchedule(para_dim, spline.IsRational(), d),
                             out,
                             QueryOptions());
  // rows in the order of SearchOrders
  const double* row = out.data();
  std::copy_n(row, dim, value);
  row += dim;
  for (int i{}; i < para_dim; ++i, row += dim) {
    std::copy_n(row, dim, first + i * dim);
  }
  if (d > 1) {
    for (int i{}; i < para_dim; ++i) {
      for (int j{i}; j < para_dim; ++j, row += dim) {
        std::copy_n(row, dim, second + (i * para_dim + j) * dim);
        std::copy_n(row, dim, second + (j * para_dim + i) * dim);
      }
    }
  }
}

void FillHessian(const splinepy::core::SplineView& spline,
                 Span<const double> at,
                 Span<double> hess) {
  auto scope = QueryArena().CreateScope();
  auto value = QueryArena().AllocateSpan<double>(spline.dim);
  auto first = QueryArena().AllocateSpan<double>(spline.para_dim * spline.dim);
  SplineDerivatives(spline, at, 2, value.data(), first.data(), hess.data());
}

} // namespace detail

using namespace detail;

void Proximity::GuessMinusQuery(Span<const double> guess,
                                Span<const double> query,
                                Span<double> difference) const {
  GuessMinusQuery(guess, query, difference, difference);
}

void Proximity::GuessMinusQuery(Span<const double> guess,
                                Span<const double> query,
                                Span<double> difference,
                                Span<double> guess_phys) const {
  splinepy::core::Evaluate(spline_,
                           Span<const double>(guess.data(), spline_.para_dim),
                           Span<double>(guess_phys.data(), spline_.dim),
                           QueryOptions());
  std::transform(guess_phys.begin(),
                 guess_phys.end(),
                 query.begin(),
                 difference.begin(),
                 std::minus<>{});
}

void Proximity::UpdateAndClip(SearchData& data) const {
  for (int i{}; i < data.para_dim; ++i) {
    data.current_guess[i] += data.delta_guess[i];
  }
  ClipGuess(data);
}

void Proximity::ComputeCostAndDerivatives(SearchData& data, int depth) const {
  // magic negative will set to the highest depth we use.
  depth = (depth < 0) ? 2 : depth;
  const int para_dim = data.para_dim;
  const int dim = data.dim;
  if (depth <= data.computed_depth
      && std::equal(data.current_guess.begin(),
                    data.current_guess.end(),
                    data.computed_at.begin())) {
    // a solver may have put back the cost of another point
    data.status.J = 0.5 * Dot(data.difference, data.difference);
    return;
  }
  std::copy(data.current_guess.begin(),
            data.current_guess.end(),
            data.computed_at.begin());
  data.computed_depth = depth;

  // value and derivatives of the spline in one call
  SplineDerivatives(spline_,
                    data.current_guess,
                    depth,
                    data.current_phys.data(),
                    data.spline_gradient.data(),
                    data.spline_hessian.data());
  std::transform(data.current_phys.begin(),
                 data.current_phys.end(),
                 data.phys_query.begin(),
                 data.difference.begin(),
                 std::minus<>{});
  data.status.J = 0.5 * Dot(data.difference, data.difference);

  if (depth > 0) {
    for (int i{}; i < para_dim; ++i) {
      // note that there's no negative here, thus not rhs yet
      data.djdu[i] =
          Dot(data.difference, data.spline_gradient.subspan(i * dim, dim));
    }
  }

  if (depth > 1) {
    ComputeGramMatrix(data);
    for (int i{}; i < para_dim * para_dim; ++i) {
      data.d2jdu2[i] =
          Dot(data.difference, data.spline_hessian.subspan(i * dim, dim))
          + data.spline_gradient_AAt[i];
    }
  }
}

void Proximity::ComputeStatus(SearchData& data,
                              const bool recompute_costs) const {
  if (recompute_costs) {
    ComputeCostAndDerivatives(data, 1);
  }

  data.status.squared_distance = Dot(data.difference, data.difference);
  data.status.distance = std::sqrt(data.status.squared_distance);
  data.status.J = 0.5 * data.status.squared_distance;
  data.status.convergence_norm = ProjectedGradientNorm(data);
}

double Proximity::ProjectedGradientNorm(const SearchData& data) const {
  double sum{};
  for (int i{}; i < data.para_dim; ++i) {
    const double g = data.djdu[i];
    if (!PushesOutOfBounds(data.current_guess[i],
                           g,
                           data.lower_bound[i],
                           data.upper_bound[i])) {
      sum += g * g;
    }
  }
  return std::sqrt(sum);
}

bool Proximity::IsAcceptable(SearchData& data, const double reference_J) const {
  if (data.status.distance < data.options.tolerance || !guard_) {
    return true;
  }

  // never end up worse than where we started
  if (data.status.J > reference_J * (1.0 + 1e-10)) {
    return false;
  }

  // second order condition on the free variables
  ComputeCostAndDerivatives(data, 2);
  MemoryArena& arena = QueryArena();
  auto scope = arena.CreateScope();
  Span<int> free = arena.AllocateSpan<int>(data.para_dim);
  int m{};
  for (int i{}; i < data.para_dim; ++i) {
    if (!PushesOutOfBounds(data.current_guess[i],
                           data.djdu[i],
                           data.lower_bound[i],
                           data.upper_bound[i])) {
      free[m++] = i;
    }
  }

  Span<double> h = arena.AllocateSpan<double>(m * m);
  for (int i{}; i < m; ++i) {
    for (int j{}; j < m; ++j) {
      h[i * m + j] = data.d2jdu2[free[i] * data.para_dim + free[j]];
    }
  }
  const double ridge = 1e-8 * std::max(MaxDiagonal(h, m), 1e-300);
  for (int i{}; i < m; ++i) {
    h[i * m + i] += ridge;
  }
  return Cholesky(h, m, 0.0);
}

} // namespace splinepy::proximity
