// Throughput benchmark of the core query kernels.
//
// Usage: core_benchmarks [--quick] [--generic] [--json <path>]
//
// --generic turns the specialized kernels off, to compare against them.
//
// Every case is warmed up once and reports the best of several repetitions as
// million queries per second (Mq/s).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/queries/detail/kernels.hpp"
#include "splinepy/core/queries/queries.hpp"

using namespace splinepy::core;

namespace {

struct Result {
  std::string kernel;
  int para_dim;
  int degree;
  bool rational;
  bool bezier;
  int nthreads;
  int64_t queries;
  double mqps;
};

/// open knot vector of degree p with n_elem uniform elements
std::vector<double> OpenKnots(int p, int n_elem) {
  std::vector<double> kv;
  for (int i = 0; i < p; ++i) {
    kv.push_back(0.0);
  }
  for (int i = 0; i <= n_elem; ++i) {
    kv.push_back(static_cast<double>(i) / n_elem);
  }
  for (int i = 0; i < p; ++i) {
    kv.push_back(1.0);
  }
  return kv;
}

SplineData MakeSpline(int para_dim,
                      int degree,
                      bool rational,
                      bool bezier,
                      std::mt19937_64& rng,
                      int n_elem = 8) {
  std::uniform_real_distribution<double> coord(-1.0, 1.0);
  std::uniform_real_distribution<double> weight(0.5, 1.5);
  SplineData d;
  d.para_dim = para_dim;
  d.dim = 3;
  d.degrees = Buffer<int32_t>(para_dim);
  int64_t n_cps = 1;
  if (!bezier) {
    d.knot_vectors = Buffer<Buffer<double>>(para_dim);
  }
  for (int a = 0; a < para_dim; ++a) {
    d.degrees[a] = degree;
    if (bezier) {
      n_cps *= degree + 1;
    } else {
      const auto kv = OpenKnots(degree, n_elem);
      d.knot_vectors[a].assign(kv.begin(), kv.end());
      n_cps *= static_cast<int64_t>(kv.size()) - degree - 1;
    }
  }
  d.control_points = Buffer<double>(n_cps * d.dim);
  for (int64_t i = 0; i < n_cps * d.dim; ++i) {
    d.control_points[i] = coord(rng);
  }
  if (rational) {
    d.weights = Buffer<double>(n_cps);
    for (int64_t i = 0; i < n_cps; ++i) {
      d.weights[i] = weight(rng);
    }
  }
  return d;
}

std::vector<double> MakeQueries(int para_dim, int64_t n, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> q(n * para_dim);
  for (auto& x : q) {
    x = u(rng);
  }
  return q;
}

/// best wall time in seconds of reps runs after one warm up run
double BestTime(const std::function<void()>& fn, int reps) {
  fn();
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  return best;
}

void Print(const Result& r) {
  std::cout << std::left << std::setw(16) << r.kernel << std::right
            << std::setw(4) << r.para_dim << std::setw(4) << r.degree
            << std::setw(5) << (r.rational ? "y" : "n") << std::setw(5)
            << (r.bezier ? "y" : "n") << std::setw(5) << r.nthreads
            << std::setw(10) << r.queries << std::setw(12) << std::fixed
            << std::setprecision(2) << r.mqps << "\n";
}

void WriteJson(const std::string& path, const std::vector<Result>& results) {
  std::ofstream f(path);
  f << "[\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    f << "  {\"kernel\": \"" << r.kernel << "\", \"para_dim\": " << r.para_dim
      << ", \"degree\": " << r.degree
      << ", \"rational\": " << (r.rational ? "true" : "false")
      << ", \"bezier\": " << (r.bezier ? "true" : "false")
      << ", \"nthreads\": " << r.nthreads << ", \"queries\": " << r.queries
      << ", \"mqps\": " << r.mqps << "}" << (i + 1 < results.size() ? "," : "")
      << "\n";
  }
  f << "]\n";
}

} // namespace

int main(int argc, char** argv) {
  bool quick = false;
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      quick = true;
    } else if (std::strcmp(argv[i], "--generic") == 0) {
      detail::KernelsEnabled() = false;
    } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
      json_path = argv[++i];
    }
  }

  const int reps = quick ? 2 : 5;
  const int64_t n_queries = quick ? (1 << 15) : (1 << 18);
  const int hw = std::max(1u, std::thread::hardware_concurrency());
  const std::vector<int> thread_counts = {1, hw};

  std::mt19937_64 rng(20261117);
  std::vector<Result> results;

  std::cout << std::left << std::setw(16) << "kernel" << std::right
            << std::setw(4) << "pd" << std::setw(4) << "p" << std::setw(5)
            << "rat" << std::setw(5) << "bez" << std::setw(5) << "thr"
            << std::setw(10) << "queries" << std::setw(12) << "Mq/s"
            << "\n";

  struct Case {
    int para_dim;
    int degree;
    bool rational;
    bool bezier;
  };
  std::vector<Case> cases;
  for (int pd = 1; pd <= 3; ++pd) {
    for (int p : {1, 2, 3, 5}) {
      if (pd == 3 && p == 5 && quick) {
        continue;
      }
      cases.push_back({pd, p, false, false});
      cases.push_back({pd, p, true, false});
    }
    cases.push_back({pd, 3, false, true});
  }

  Options base;
  base.validate = false;

  for (const auto& c : cases) {
    const SplineData data =
        MakeSpline(c.para_dim, c.degree, c.rational, c.bezier, rng);
    const SplineView view = data.View();
    const auto queries = MakeQueries(c.para_dim, n_queries, rng);
    const Span<const double> q(queries.data(), queries.size());
    const int64_t n_supports = view.NumSupports();

    Buffer<double> out(n_queries * 3 * c.para_dim);
    Buffer<double> basis(n_queries * n_supports);
    Buffer<int32_t> support(n_queries * n_supports);
    std::vector<int32_t> first_order(c.para_dim, 0);
    first_order[0] = 1;
    std::vector<int32_t> mixed_order(c.para_dim, 1);
    // gradient and the mixed second derivative, as for a Hessian row
    std::vector<int32_t> several_orders;
    for (int a = 0; a < c.para_dim; ++a) {
      for (int b = 0; b < c.para_dim; ++b) {
        several_orders.push_back(a == b ? 1 : 0);
      }
    }
    several_orders.insert(several_orders.end(),
                          mixed_order.begin(),
                          mixed_order.end());
    const DerivativeOrders schedule(
        Span<const int32_t>(several_orders.data(), several_orders.size()),
        c.para_dim,
        c.rational);
    Buffer<double> several_out(n_queries * (c.para_dim + 1) * 3);

    for (int thr : thread_counts) {
      Options opt = base;
      opt.nthreads = thr;
      const auto run = [&](const std::string& name,
                           const std::function<void()>& fn) {
        const double t = BestTime(fn, reps);
        Result r{name,
                 c.para_dim,
                 c.degree,
                 c.rational,
                 c.bezier,
                 thr,
                 n_queries,
                 static_cast<double>(n_queries) / t * 1e-6};
        Print(r);
        results.push_back(r);
      };

      run("evaluate", [&] { Evaluate(view, q, out.span(), opt); });
      run("jacobian", [&] { Jacobian(view, q, out.span(), opt); });
      run("derivative_1", [&] {
        Derivative(view,
                   q,
                   Span<const int32_t>(first_order.data(), first_order.size()),
                   out.span(),
                   opt);
      });
      if (c.para_dim > 1) {
        run("derivative_mix", [&] {
          Derivative(
              view,
              q,
              Span<const int32_t>(mixed_order.data(), mixed_order.size()),
              out.span(),
              opt);
        });
      }
      run("basis_support",
          [&] { BasisAndSupport(view, q, basis.span(), support.span(), opt); });
      run("basis_der_1", [&] {
        BasisDerivativeAndSupport(
            view,
            q,
            Span<const int32_t>(first_order.data(), first_order.size()),
            basis.span(),
            support.span(),
            opt);
      });
      run("derivatives",
          [&] { Derivative(view, q, schedule, several_out.span(), opt); });
    }
  }

  // many small patches, as in microstructures
  {
    const int n_patches = quick ? 128 : 1024;
    const int64_t per_patch = 256;
    std::vector<SplineData> patches;
    patches.reserve(n_patches);
    for (int i = 0; i < n_patches; ++i) {
      patches.push_back(MakeSpline(3, 2, false, true, rng));
    }
    std::vector<SplineView> views;
    for (const auto& p : patches) {
      views.push_back(p.View());
    }
    const auto queries = MakeQueries(3, per_patch, rng);
    const Span<const double> q(queries.data(), queries.size());
    Buffer<double> out(n_patches * per_patch * 3);
    std::vector<Span<double>> outs;
    for (int i = 0; i < n_patches; ++i) {
      outs.push_back(out.span().subspan(i * per_patch * 3, per_patch * 3));
    }
    for (int thr : thread_counts) {
      Options opt = base;
      opt.nthreads = thr;
      const double t = BestTime(
          [&] {
            Evaluate(SplineBatchView(views.data(), views.size()),
                     Span<const Span<const double>>(&q, 1),
                     Span<Span<double>>(outs.data(), outs.size()),
                     opt);
          },
          reps);
      Result r{"batch_evaluate",
               3,
               2,
               false,
               true,
               thr,
               n_patches * per_patch,
               static_cast<double>(n_patches * per_patch) / t * 1e-6};
      Print(r);
      results.push_back(r);
    }
  }

  if (!json_path.empty()) {
    WriteJson(json_path, results);
    std::cout << "wrote " << json_path << "\n";
  }
  return 0;
}
