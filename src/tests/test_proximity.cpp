#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/proximity/proximity.hpp"

namespace splinepy::proximity {
namespace {

/// bilinear surface with the edge u=1 collapsed to the apex of a cone
struct Cone {
  std::vector<int32_t> degrees{1, 1};
  std::vector<double> kv{0, 0, 1, 1};
  std::vector<splinepy::core::Span<const double>> kv_spans;
  std::vector<double> cps{0, 0, 0, 1, .5, 1, 0, 1, 0, 1, .5, 1};

  Cone() {
    kv_spans = {splinepy::core::Span<const double>(kv),
                splinepy::core::Span<const double>(kv)};
  }

  splinepy::core::SplineView View() const {
    splinepy::core::SplineView v;
    v.para_dim = 2;
    v.dim = 3;
    v.degrees = splinepy::core::Span<const int32_t>(degrees);
    v.knot_vectors =
        splinepy::core::Span<const splinepy::core::Span<const double>>(
            kv_spans.data(),
            2);
    v.control_points_data = cps.data();
    v.num_control_points = 4;
    v.cp_stride = 3;
    v.dim_stride = 1;
    return v;
  }
};

double Distance(const splinepy::core::SplineView& view,
                const double* u,
                const double* query) {
  double x[3];
  splinepy::core::Evaluate(view,
                           splinepy::core::Span<const double>(u, 2),
                           splinepy::core::Span<double>(x, 3));
  double d2{};
  for (int i = 0; i < 3; ++i) {
    d2 += (x[i] - query[i]) * (x[i] - query[i]);
  }
  return std::sqrt(d2);
}

TEST(Proximity, BvhFindsClosestPointNextToCollapsedEdge) {
  Cone cone;
  const auto view = cone.View();
  Proximity proximity(view);
  const int resolutions[2] = {4, 4};
  ASSERT_TRUE(proximity.PlantBvh(resolutions));

  std::mt19937 gen(5);
  std::uniform_real_distribution<double> ux(0.9, 1.3), uy(-0.4, 0.), uz(.7, 1.);
  for (int n = 0; n < 30; ++n) {
    const double query[3] = {ux(gen), uy(gen), uz(gen)};
    double guess[2], nearest[3], diff[3], dist{}, conv{}, first[6], second[12];
    proximity.VerboseQuery(query,
                           1e-10,
                           100,
                           false,
                           guess,
                           nearest,
                           diff,
                           dist,
                           conv,
                           first,
                           second);

    // brute force reference
    double best = 1e100;
    for (int i = 0; i <= 120; ++i) {
      for (int j = 0; j <= 120; ++j) {
        const double u[2] = {i / 120., j / 120.};
        best = std::min(best, Distance(view, u, query));
      }
    }
    EXPECT_LE(dist, best + 1e-3);
  }
}

TEST(Proximity, SearchStepBoundsTheSearchAroundItsStart) {
  Cone cone;
  const auto view = cone.View();
  const Proximity proximity(view);
  const double target[2] = {.8, .7};
  double query[3];
  splinepy::core::Evaluate(view,
                           splinepy::core::Span<const double>(target, 2),
                           splinepy::core::Span<double>(query, 3));
  const double start[2] = {.2, .2};
  const double step[2] = {.1, .25};
  for (const double* search_step :
       {static_cast<const double*>(nullptr), step}) {
    double guess[2], nearest[3], diff[3], dist{}, conv{}, first[6];
    proximity.SolveFrom(query,
                        1e-10,
                        100,
                        search_step,
                        start,
                        guess,
                        nearest,
                        diff,
                        dist,
                        conv,
                        first,
                        nullptr);
    if (search_step) {
      // the closest point of the box around the start is its corner
      EXPECT_DOUBLE_EQ(guess[0], .3);
      EXPECT_DOUBLE_EQ(guess[1], .45);
    } else {
      EXPECT_NEAR(guess[0], target[0], 1e-8);
      EXPECT_NEAR(guess[1], target[1], 1e-8);
    }
  }
}

TEST(Proximity, IndividualSolversOnEmbeddedCurveWithOptionalOutputs) {
  const int32_t degrees[] = {1};
  const double points[] = {0, 0, 0, 2, 0, 0};
  splinepy::core::SplineView view;
  view.para_dim = 1;
  view.dim = 3;
  view.degrees = {degrees, 1};
  view.control_points_data = points;
  view.num_control_points = 2;
  view.cp_stride = 3;
  view.dim_stride = 1;

  const int resolution[] = {4};
  const double query[] = {0.7, 0.2, 0.3};
  for (const auto solver : {Solver::kNewton,
                            Solver::kDogleg,
                            Solver::kSlsqp,
                            Solver::kLevenbergMarquart}) {
    Proximity proximity(view);
    proximity.SetSolverChain({&solver, 1});
    ASSERT_TRUE(proximity.PlantBvh(resolution));
    double parameter{}, distance{}, convergence{};
    proximity.VerboseQuery(query,
                           1e-10,
                           100,
                           false,
                           &parameter,
                           nullptr,
                           nullptr,
                           distance,
                           convergence,
                           nullptr,
                           nullptr);
    EXPECT_NEAR(parameter, 0.35, 1e-7);
    EXPECT_NEAR(distance, std::sqrt(0.13), 1e-10);
  }
}

} // namespace
} // namespace splinepy::proximity
