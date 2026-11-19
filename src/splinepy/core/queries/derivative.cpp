#include "splinepy/core/queries/derivative.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <map>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/detail/kernels.hpp"
#include "splinepy/core/queries/detail/query_helpers.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

void Derivative(const SplineView& spline,
                Span<const double> queries,
                Span<const int32_t> orders,
                Span<double> out,
                const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline, options);
  }
  detail::CheckSplineDimensions(spline);
  if (detail::CheckDerivativeOrders(spline, orders)) {
    Evaluate(spline, queries, out, options);
    return;
  }
  const int64_t num_queries = detail::NumQueries(spline, queries);
  if (num_queries == 0) {
    return;
  }

  const int32_t para_dim = spline.para_dim;
  const int32_t dim = spline.dim;
  const int64_t expected_out_size = num_queries * dim;
  detail::CheckOutputSize(
      out,
      expected_out_size,
      "Output span size is smaller than expected (num_queries * dim)");

  // polynomials vanish if any order exceeds the degree
  if (!spline.IsRational()) {
    for (int32_t d = 0; d < para_dim; ++d) {
      if (orders[d] > spline.degrees[d]) {
        std::fill_n(out.data(), expected_out_size, 0.0);
        return;
      }
    }
  }

  if (detail::FastKernelsApply(spline)) {
    int32_t lo[detail::kKernelMaxParaDim];
    for (int32_t d = 0; d < para_dim; ++d) {
      lo[d] = spline.IsRational() ? 0 : orders[d];
    }
    if (detail::KernelTermCount(para_dim, lo, orders.data()) > 0) {
      detail::DispatchKernel(spline, [&](auto pd, auto rational, auto knots) {
        static constexpr int kPd = decltype(pd)::value;
        ParallelFor(int64_t{0}, num_queries, options.nthreads, [&](int64_t q) {
          detail::DerivativeQuery<kPd,
                                  decltype(rational)::value,
                                  decltype(knots)::value>(spline,
                                                          queries.data()
                                                              + q * kPd,
                                                          orders.data(),
                                                          out.data() + q * dim);
        });
      });
      return;
    }
  }

  // beyond the kernel limits: the generic multi-order path
  Options generic = options;
  generic.validate = false;
  Derivative(spline,
             queries,
             DerivativeOrders(orders, para_dim, spline.IsRational()),
             out,
             generic);
}

DerivativeOrders::DerivativeOrders(Span<const int32_t> orders,
                                   std::size_t para_dim,
                                   bool rational)
    : para_dim_(para_dim),
      rational_(rational) {
  if (para_dim == 0 || orders.empty() || orders.size() % para_dim) {
    throw SplinepyException("Invalid derivative multi-indices");
  }
  axis_orders_.resize(para_dim);
  // orders padded with zeros beyond para_dim, sorted lexicographically
  using Order = std::array<int32_t, kMaxParaDim>;
  const auto order_of = [&](Span<const int32_t> values) {
    Order padded{};
    std::copy(values.begin(), values.end(), padded.begin());
    return padded;
  };
  std::map<Order, std::size_t> indices;
  const auto count = orders.size() / para_dim;
  for (std::size_t i = 0; i < count; ++i) {
    const auto order = orders.subspan(i * para_dim, para_dim);
    for (std::size_t axis = 0; axis < para_dim; ++axis) {
      if (order[axis] < 0) {
        throw SplinepyException("Derivative orders must be non-negative");
      }
      axis_orders_[axis] = std::max(axis_orders_[axis], order[axis]);
    }
    if (rational) {
      // The union of the requested componentwise lower orders is closed
      // under subtraction of any sub-order.
      Order sub{};
      for (;;) {
        indices.try_emplace(sub, 0);
        std::size_t axis = 0;
        while (axis < para_dim && sub[axis] == order[axis]) {
          sub[axis++] = 0;
        }
        if (axis == para_dim) {
          break;
        }
        ++sub[axis];
      }
    } else {
      indices.try_emplace(order_of(order), 0);
    }
  }
  terms_ = Buffer<int32_t>(indices.size() * para_dim);
  std::size_t term = 0;
  for (auto& [order, index] : indices) {
    index = term;
    std::copy_n(order.begin(), para_dim, terms_.data() + term++ * para_dim);
  }
  output_indices_ = Buffer<std::size_t>(count);
  for (std::size_t i = 0; i < count; ++i) {
    output_indices_[i] =
        indices.at(order_of(orders.subspan(i * para_dim, para_dim)));
  }
}

void Derivative(const SplineView& spline,
                Span<const double> queries,
                const DerivativeOrders& orders,
                Span<double> out,
                const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline, options);
  }
  const auto pd = orders.para_dim_;
  if (spline.para_dim != static_cast<int64_t>(pd) || spline.dim <= 0
      || spline.IsRational() != orders.rational_ || queries.size() % pd) {
    throw SplinepyException("Derivative plan does not match spline or queries");
  }
  const auto dim = static_cast<std::size_t>(spline.dim);
  const auto count = queries.size() / pd;
  const auto output_count = orders.output_indices_.size();
  if (count == 0) {
    return;
  }
  if (count > out.size() / dim / output_count) {
    throw SplinepyException("Derivative output span is too small");
  }

  if (detail::FastKernelsApply(spline)) {
    // one contraction over all orders up to the highest one per axis
    const int32_t lo[detail::kKernelMaxParaDim] = {};
    int32_t hi[detail::kKernelMaxParaDim] = {};
    for (std::size_t axis = 0; axis < pd; ++axis) {
      hi[axis] = orders.axis_orders_[axis];
    }
    if (detail::KernelTermCount(static_cast<int32_t>(pd), lo, hi) > 0) {
      Buffer<int32_t> requested(output_count * pd);
      for (std::size_t i = 0; i < output_count; ++i) {
        std::copy_n(orders.terms_.data() + orders.output_indices_[i] * pd,
                    pd,
                    requested.data() + i * pd);
      }
      // terms_ of a rational schedule hold all lower orders of the requests
      bool needed[detail::kKernelMaxTerms] = {};
      for (std::size_t t = 0; t < orders.terms_.size() / pd; ++t) {
        int32_t index = 0;
        int32_t stride = 1;
        for (std::size_t axis = 0; axis < pd; ++axis) {
          index += orders.terms_[t * pd + axis] * stride;
          stride *= hi[axis] + 1;
        }
        needed[index] = true;
      }
      const auto n_out = static_cast<int32_t>(output_count);
      detail::DispatchKernel(spline, [&](auto kpd, auto rational, auto knots) {
        static constexpr int kPd = decltype(kpd)::value;
        ParallelFor(std::size_t{0},
                    count,
                    options.nthreads,
                    [&](std::size_t q) {
                      detail::DerivativesQuery<kPd,
                                               decltype(rational)::value,
                                               decltype(knots)::value>(
                          spline,
                          queries.data() + q * kPd,
                          hi,
                          n_out,
                          requested.data(),
                          orders.rational_ ? needed : nullptr,
                          out.data() + q * output_count * dim);
                    });
      });
      return;
    }
  }

  const auto term_count = orders.terms_.size() / pd;
  if (term_count > SIZE_MAX / sizeof(double) / dim) {
    throw SplinepyException("Derivative scratch exceeds addressable size");
  }
  const Span<const int32_t> terms(orders.terms_.data(), orders.terms_.size());
  // Find terms by their componentwise indices without materializing a dense
  // grid spanning unrelated requests.
  const auto find_term = [&](Span<const int32_t> order, std::size_t end) {
    std::size_t first = 0;
    while (first < end) {
      const auto middle = first + (end - first) / 2;
      const auto candidate = terms.subspan(middle * pd, pd);
      if (std::lexicographical_compare(candidate.begin(),
                                       candidate.end(),
                                       order.begin(),
                                       order.end())) {
        first = middle + 1;
      } else {
        end = middle;
      }
    }
    return first;
  };
  const auto threads =
      std::min(count,
               static_cast<std::size_t>(ResolveThreadCount(options.nthreads)));
  WorkspaceRef workspace(options,
                         MemoryArena::kDefaultBlockSize,
                         static_cast<int>(threads));
  const auto cp_strides = ControlGrid(spline).stride;
  ParallelForWithThreadId(
      std::size_t{0},
      count,
      options.nthreads,
      [&](std::size_t q, int thread) {
        auto& arena = workspace->GetThreadArena(thread);
        auto scope = arena.CreateScope();
        auto starts = arena.AllocateSpan<int32_t>(pd);
        auto axis_offsets = arena.AllocateSpan<std::size_t>(pd + 1);
        axis_offsets[0] = 0;
        for (std::size_t axis = 0; axis < pd; ++axis) {
          axis_offsets[axis + 1] =
              axis_offsets[axis]
              + (std::size_t{1}
                 + std::min(orders.axis_orders_[axis], spline.degrees[axis]))
                    * (std::size_t{1} + spline.degrees[axis]);
        }
        auto tables = arena.AllocateSpan<double>(axis_offsets[pd]);
        for (std::size_t axis = 0; axis < pd; ++axis) {
          int32_t span;
          detail::Evaluate1DAxisBasis(
              spline,
              axis,
              queries[q * pd + axis],
              std::min(orders.axis_orders_[axis], spline.degrees[axis]),
              span,
              starts[axis],
              tables.subspan(axis_offsets[axis],
                             axis_offsets[axis + 1] - axis_offsets[axis]));
        }

        // value(j) of every support, axis 0 fastest, expanded one axis at a
        // time from the last: out[k * width + j] = combine(out[k], axis, j)
        const auto supports = spline.NumSupports();
        const auto expand = [&](auto* out, auto one, auto&& combine) {
          int64_t size = 1;
          out[0] = one;
          for (std::size_t axis = pd; axis-- > 0;) {
            const int32_t width = spline.degrees[axis] + 1;
            for (int64_t k = size; k-- > 0;) {
              const auto outer = out[k];
              for (int32_t j = width; j-- > 0;) {
                out[k * width + j] = combine(outer, axis, j);
              }
            }
            size *= width;
          }
        };
        auto points = arena.AllocateSpan<int64_t>(supports);
        expand(points.data(),
               int64_t{0},
               [&](int64_t outer, auto axis, int32_t j) {
                 return outer + (starts[axis] + j) * cp_strides[axis];
               });
        auto basis = arena.AllocateSpan<double>(supports);

        auto values = arena.AllocateSpan<double>(term_count * dim, 0.0);
        auto weights =
            arena.AllocateSpan<double>(orders.rational_ ? term_count : 0);
        auto difference =
            arena.AllocateSpan<int32_t>(orders.rational_ ? pd : 0);
        auto gamma = arena.AllocateSpan<int32_t>(orders.rational_ ? pd : 0);
        auto correction =
            arena.AllocateSpan<double>(orders.rational_ ? dim : 0);
        for (std::size_t i = 0; i < term_count; ++i) {
          const auto order = terms.subspan(i * pd, pd);
          auto value = values.subspan(i * dim, dim);
          double weight = 0.0;
          if (std::equal(
                  order.begin(),
                  order.end(),
                  spline.degrees.begin(),
                  [](int32_t k, int32_t degree) { return k <= degree; })) {
            expand(basis.data(), 1.0, [&](double outer, auto axis, int32_t j) {
              const auto width = std::size_t(spline.degrees[axis] + 1);
              return outer
                     * tables[axis_offsets[axis] + order[axis] * width + j];
            });
            if (orders.rational_) {
              for (int64_t support = 0; support < supports; ++support) {
                basis[support] *= spline.Weight(points[support]);
                weight += basis[support];
              }
            }
            // one component at a time, so the sum stays in a register
            for (std::size_t component = 0; component < dim; ++component) {
              const double* x =
                  spline.control_points_data + component * spline.dim_stride;
              double sum = 0.0;
              for (int64_t support = 0; support < supports; ++support) {
                sum += basis[support] * x[points[support] * spline.cp_stride];
              }
              value[component] = sum;
            }
          }
          if (!orders.rational_) {
            continue;
          }
          weights[i] = weight;
          std::fill(correction.begin(), correction.end(), 0.0);
          std::fill(gamma.begin(), gamma.end(), 0);
          for (;;) {
            // Higher derivatives of the polynomial weight spline vanish once
            // any component exceeds the degree.
            std::size_t axis = 0;
            while (axis < pd
                   && gamma[axis]
                          == std::min(order[axis], spline.degrees[axis])) {
              gamma[axis++] = 0;
            }
            if (axis == pd) {
              break;
            }
            ++gamma[axis];
            double factor = weights[find_term(gamma, i + 1)];
            if (factor == 0.0) {
              continue;
            }
            for (axis = 0; axis < pd; ++axis) {
              difference[axis] = order[axis] - gamma[axis];
              factor *= BinomialDouble(order[axis], gamma[axis]);
            }
            const auto sub = find_term(difference, i);
            for (std::size_t component = 0; component < dim; ++component) {
              correction[component] += factor * values[sub * dim + component];
            }
          }
          const auto inverse_weight =
              weights[0] != 0.0 ? 1.0 / weights[0] : 0.0;
          for (std::size_t component = 0; component < dim; ++component) {
            value[component] =
                (value[component] - correction[component]) * inverse_weight;
          }
        }
        for (std::size_t i = 0; i < output_count; ++i) {
          std::copy_n(values.data() + orders.output_indices_[i] * dim,
                      dim,
                      out.data() + (q * output_count + i) * dim);
        }
      });
}

Buffer<double> Derivative(const SplineView& spline,
                          Span<const double> queries,
                          Span<const int32_t> orders,
                          const Options& options) {
  detail::CheckSplineDimensions(spline);
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  Buffer<double> out(num_queries * spline.dim);
  Derivative(spline, queries, orders, out.span(), options);
  return out;
}

void Derivative(SplineBatchView batch,
                Span<const Span<const double>> queries_per_spline,
                Span<const int32_t> orders,
                Span<Span<double>> out_per_spline,
                const Options& options) {
  detail::ForEachInBatch(
      batch,
      queries_per_spline,
      out_per_spline,
      options,
      [&](const SplineView& s) {
        const auto pd = static_cast<std::size_t>(s.para_dim);
        const auto sets = orders.size() <= pd ? 1 : orders.size() / pd;
        return static_cast<int64_t>(sets) * s.dim;
      },
      [&](const SplineView& spline,
          Span<const double> queries,
          Span<double> out,
          const Options& patch_options) {
        if (orders.size() <= static_cast<std::size_t>(spline.para_dim)) {
          Derivative(spline, queries, orders, out, patch_options);
        } else if (!queries.empty()) {
          const DerivativeOrders schedule(orders,
                                          spline.para_dim,
                                          spline.IsRational());
          Derivative(spline, queries, schedule, out, patch_options);
        }
      });
}

} // namespace splinepy::core
