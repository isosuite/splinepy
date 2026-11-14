#include <gtest/gtest.h>

#include <vector>

#include "splinepy/core/validate/validator.hpp"

using namespace splinepy::core;

TEST(ValidatorTest, ValidBezierView) {
  int32_t degrees[2] = {2, 1};
  double cps[6 * 2] = {
      0.0,
      0.0,
      1.0,
      0.0,
      2.0,
      0.0,
      0.0,
      1.0,
      1.0,
      1.0,
      2.0,
      1.0,
  };

  SplineView v;
  v.para_dim = 2;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees, 2);
  v.control_points_data = cps;
  v.num_control_points = 6;
  v.cp_stride = 2;
  v.dim_stride = 1;

  EXPECT_EQ(Validate(v), Status::kOk);
  EXPECT_NO_THROW(ThrowIfInvalid(v));
}

TEST(ValidatorTest, ValidBSplineView) {
  int32_t degrees[1] = {2};
  double kv0[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  Span<const double> kvs[1] = {Span<const double>(kv0, 6)};
  double cps[3 * 2] = {0.0, 0.0, 1.0, 2.0, 2.0, 0.0};

  SplineView v;
  v.para_dim = 1;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees, 1);
  v.knot_vectors = Span<const Span<const double>>(kvs, 1);
  v.control_points_data = cps;
  v.num_control_points = 3;
  v.cp_stride = 2;
  v.dim_stride = 1;

  EXPECT_EQ(Validate(v), Status::kOk);
  EXPECT_NO_THROW(ThrowIfInvalid(v));
}

TEST(ValidatorTest, InvalidDecreasingKnotVector) {
  int32_t degrees[1] = {1};
  double bad_kv[4] = {0.0, 1.0, 0.5, 1.0}; // not non-decreasing
  Span<const double> kvs[1] = {Span<const double>(bad_kv, 4)};
  double cps[2 * 2] = {0.0, 0.0, 1.0, 1.0};

  SplineView v;
  v.para_dim = 1;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees, 1);
  v.knot_vectors = Span<const Span<const double>>(kvs, 1);
  v.control_points_data = cps;
  v.num_control_points = 2;
  v.cp_stride = 2;
  v.dim_stride = 1;

  EXPECT_EQ(Validate(v), Status::kInvalidArgument);
  EXPECT_THROW(ThrowIfInvalid(v), SplinepyException);
}

TEST(ValidatorTest, InvalidControlPointCount) {
  int32_t degrees[1] = {1};
  double cps[4] = {0.0, 0.0, 1.0, 1.0}; // 2 points

  SplineView v;
  v.para_dim = 1;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees, 1);
  v.control_points_data = cps;
  v.num_control_points = 5; // Expected 2!
  v.cp_stride = 2;
  v.dim_stride = 1;

  EXPECT_EQ(Validate(v), Status::kDimensionMismatch);
  EXPECT_THROW(ThrowIfInvalid(v), SplinepyException);
}

TEST(ValidatorTest, InvalidNegativeWeight) {
  int32_t degrees[1] = {1};
  double cps[4] = {0.0, 0.0, 1.0, 1.0};
  double weights[2] = {1.0, -0.5}; // negative weight

  SplineView v;
  v.para_dim = 1;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees, 1);
  v.control_points_data = cps;
  v.num_control_points = 2;
  v.cp_stride = 2;
  v.dim_stride = 1;
  v.weights_data = weights;
  v.weight_stride = 1;

  EXPECT_EQ(Validate(v), Status::kInvalidArgument);
  EXPECT_THROW(ThrowIfInvalid(v), SplinepyException);
}
