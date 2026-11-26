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

#pragma once

#include <algorithm>
#include <memory>
#include <utility>

#include "splinepy/core/memory/buffer.hpp"

#include "splinepy/core/error.hpp"
#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/spatial/element_bvh.hpp"

namespace splinepy::proximity {

/// Validate the base iteration budget before applying fallback multipliers.
void ValidateIterationBudget(int max_iterations);

using splinepy::core::Span;

/// @brief Source of a single starting point for the local solvers, e.g. the
/// nearest sample of a point cloud. Implementations may depend on libraries
/// that the core does not have (see grid_guess.hpp).
class InitialGuess {
public:
  virtual ~InitialGuess() = default;

  /// @brief writes a parametric guess (para_dim) for a physical query (dim)
  virtual void Guess(const double* query, double* guess) const = 0;
};

namespace detail {
struct SearchData;
}

/// local solvers that can be chained in Proximity::VerboseQuery.
enum class Solver { kNewton, kDogleg, kSlsqp, kLevenbergMarquart };

/*!
 * A helper class to perform proximity operations for splines.
 *
 * Given a physical coordinate, this tries to find parametric coordinate that
 * maps to the closest physical coordinate. Often referred as
 * "point inversion". This is done by searching for a root of first derivative
 * of squared distance between mapped coordinate and query coordinate.
 * For detailed information, please take a look at splinepy python
 * documentation.
 *
 * J = 1 / 2 * (p - S(u))^2
 */
class Proximity {
public:
  // helpee spline
  splinepy::core::SplineView spline_;

  // initial guess source used if no bvh is planted
  std::unique_ptr<InitialGuess> initial_guess_;

  // element bvh based initial guesses
  std::unique_ptr<splinepy::core::ElementBvh> bvh_;
  splinepy::core::AxisArray<int> bvh_samples_per_axis_;
  /// spline(sample) of every element's sampling grid, element-major, first
  /// parametric dimension fastest. Empty if it would exceed the memory budget,
  /// in which case samples are evaluated at query time.
  splinepy::core::Buffer<double> bvh_sample_points_;

  /// at most this many starting points are tried per query
  int max_starts_{4};
  /// at most this many elements are sampled per query
  int max_bvh_visits_{64};

  /// solvers are tried in this order, until one of them returns an acceptable
  /// result
  splinepy::core::Buffer<Solver> solver_chain_{Solver::kDogleg,
                                               Solver::kSlsqp,
                                               Solver::kLevenbergMarquart};

  /// if true, a result is only accepted if it is not worse than the guess it
  /// started from, and if it is a (semi-)definite minimum within the bounds.
  bool guard_{true};

public:
  /// Constructor. As a spline helper class, always need a spline.
  explicit Proximity(splinepy::core::SplineView spline) : spline_(spline) {}

  /// @brief sets the order of solvers. Unknown/empty chains are not allowed.
  void SetSolverChain(Span<const Solver> chain) {
    if (chain.empty()) {
      splinepy::core::ThrowError(splinepy::core::Status::kInvalidArgument,
                                 "solver chain can't be empty");
    }
    solver_chain_.assign(chain.begin(), chain.end());
  }

  void SetGuard(const bool guard) { guard_ = guard; }

  /// @brief Refreshes the view of the spline. The view refers to spans that
  /// the caller may have rebuilt (e.g. knot vector spans), so call this before
  /// each query session. Data (control points, ...) must be unchanged.
  void SetSpline(splinepy::core::SplineView spline) { spline_ = spline; }

  void SetMaxStarts(const int max_starts) {
    max_starts_ = std::max(1, max_starts);
  }

  /// @brief Builds a bounding volume hierarchy over the knot spans of the
  /// spline, as an alternative to the kd-tree. Elements are sampled lazily at
  /// query time. The total resolution is distributed to the elements, so that
  /// the number of samples is comparable to a kd-tree with the same
  /// resolutions. Returns false (and plants nothing) if the bounding boxes
  /// are not valid, e.g. due to non-positive weights.
  /// @param resolutions parameter space sampling resolution
  bool PlantBvh(const int* resolutions);

  bool HasBvh() const { return static_cast<bool>(bvh_); }

private:
  struct StartView {
    double lower_bound_squared;
    double distance_squared;
    Span<const double> guess;
    Span<const double> step;
  };

  Span<StartView> CollectBvhStarts(const double* query,
                                   splinepy::core::MemoryArena& arena) const;

public:
  /// @brief Sets the initial guess source for queries without a bvh.
  void SetInitialGuess(std::unique_ptr<InitialGuess> initial_guess) {
    initial_guess_ = std::move(initial_guess);
  }

  bool HasInitialGuess() const { return static_cast<bool>(initial_guess_); }

  /// @brief difference = spline(guess) - query. In current formulation, this is
  /// our objective function.
  /// @param guess
  /// @param query
  /// @param difference
  void GuessMinusQuery(Span<const double> guess,
                       Span<const double> query,
                       Span<double> difference) const;

  /// @brief difference = spline(guess) - query.
  /// This returns spline(guess)
  /// @param guess
  /// @param query
  /// @param difference
  /// @param guess_phys
  void GuessMinusQuery(Span<const double> guess,
                       Span<const double> query,
                       Span<double> difference,
                       Span<double> guess_phys) const;

  /// @brief make initial guess from the initial guess source
  /// @param goal
  /// @param guess
  void MakeInitialGuess(Span<const double> goal, Span<double> guess) const;

private:
  /// @brief updates current_guess with delta_guess and clips result
  /// @param data
  void UpdateAndClip(detail::SearchData& data) const;

  /// @brief computes J and its derivatives based on the value of depth. depth
  /// == 1 -> jacobian, dept > 1 -> jacobian & hessian. Negative value is
  /// considered as highest depth we can compute (currently hessian)
  /// @param data
  /// @param depth
  void ComputeCostAndDerivatives(detail::SearchData& data, int depth) const;

  /// @brief Computes status in SearchData based on current values in
  /// SearchData. If recompute_costs==true, it will compute the cost and its
  /// derivative again and store it.
  /// @param data
  /// @param recompute_costs
  void ComputeStatus(detail::SearchData& data,
                     const bool recompute_costs) const;

  /// @brief resets search bounds to spline's parametric bounds
  /// @param data
  void FindSearchBound(detail::SearchData& data) const;

  /// @brief Prepares lhs and rhs of newton iteration
  void PrepareIterationNewton(detail::SearchData& data) const;

  /// @brief Performs newton iterations using PrepareIterationNewton
  /// @param data
  void Newton(detail::SearchData& data) const;

  /// @brief Prepares lhs and rhs of LM iteration
  /// @param data
  void PrepareIterationLevenbergMarquart(detail::SearchData& data) const;

  /// @brief Levenberg-Marquart method. Intended to be used as a fallback method
  /// for newton. Uses PrepareIterationLevenbergMarquart()
  /// @param data
  void LevenbergMarquart(detail::SearchData& data) const;

  /// @brief norm of the gradient of J, after removing the components that
  /// point out of the search bounds. This vanishes at constrained minima.
  double ProjectedGradientNorm(const detail::SearchData& data) const;

  /// @brief Trust region method with dogleg steps. Uses the exact hessian if it
  /// is positive definite, else the Gauss-Newton matrix. Respects search
  /// bounds by projecting steps and freezing variables that push out.
  /// @param data
  void Dogleg(detail::SearchData& data) const;

  /// @brief is the current result acceptable? (see guard_). Fills the
  /// hessian of J and of the spline at the current guess as a byproduct.
  /// @param data
  /// @param reference_J J of the guess the solver started from
  bool IsAcceptable(detail::SearchData& data, const double reference_J) const;

  /// @brief Prepares lhs and rhs of slsqp iteration and other parameters
  void PrepareIterationSlsqp(detail::SearchData& data) const;

  /// @brief Performs slsqp iterations using PrepareIterationSlsqp
  /// @param data
  void Slsqp(detail::SearchData& data) const;

public:
  /// @brief Runs the solver chain from a given start. Same outputs as
  /// VerboseQuery().
  void SolveFrom(const double* query,
                 const double& tolerance,
                 const int& max_iterations,
                 const double* start,
                 double* final_guess,
                 double* nearest,
                 double* nearest_minus_query,
                 double& distance,
                 double& convergence_norm,
                 double* first_derivatives,
                 double* second_derivatives) const;

  /// @brief Given physical coordinate, finds closest parametric coordinate.
  /// Takes initial guess from the initial guess source, or starts from several
  /// promising elements if a bvh was planted.
  ///
  /// @param[in] query
  /// @param[in] tolerance
  /// @param[in] max_iterations
  /// @param[out] final_guess (para_dim)
  /// @param[out] nearest (dim)
  /// @param[out] nearest_minus_query (dim)
  /// @param[out] distance
  /// @param[out] convergence_norm
  /// @param[out] first_derivatives (para_dim x dim)
  /// @param[out] second_derivatives (para_dim x para_dim x dim)
  void VerboseQuery(const double* query,
                    const double& tolerance,
                    const int& max_iterations,
                    double* final_guess,
                    double* nearest /* spline(final_guess) */,
                    double* nearest_minus_query /* difference */,
                    double& distance,
                    double& convergence_norm,
                    double* first_derivatives /* spline jacobian */,
                    double* second_derivatives /* spline hessian */) const;
};

} // namespace splinepy::proximity
