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
#include <vector>

#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/spatial/element_bvh.hpp"
#include "splinepy/utils/arrays.hpp"

namespace splinepy::proximity {

using RealArray = splinepy::utils::Array<double>;
using RealArray2D = splinepy::utils::Array<double, 2>;
using RealArray3D = splinepy::utils::Array<double, 3>;
using IndexArray = splinepy::utils::Array<int>;
using ConstRealArray = splinepy::utils::Array<const double>;
using LhsMatrix = splinepy::utils::Matrix<double, int>;

/// @brief Source of a single starting point for the local solvers, e.g. the
/// nearest sample of a point cloud. Implementations may depend on libraries
/// that the core does not have (see grid_guess.hpp).
class InitialGuess {
public:
  virtual ~InitialGuess() = default;

  /// @brief writes a parametric guess (para_dim) for a physical query (dim)
  virtual void Guess(const double* query, double* guess) const = 0;

  /// @brief parametric distance between samples per axis (para_dim), used to
  /// bound the search in the neighborhood of the guess. Null if unknown.
  virtual const double* StepSize() const { return nullptr; }
};

/// @brief compute convergence goal based on first norm and tolerances.
/// @tparam T
/// @param first_norm
/// @param rel_tol
/// @param abs_tol
/// @return
template<typename T>
T ConvergenceGoal(const T& first_norm, const T& rel_tol, const T& abs_tol) {
  return std::max(first_norm * rel_tol, abs_tol);
}

/// struct to hold temporary data during the search.
/// This should allow easier transition to fallback methods.
struct SearchData {
  int para_dim;
  int dim;

  // view array
  ConstRealArray phys_query;
  // either view or allocated
  RealArray current_guess;
  RealArray current_phys;
  RealArray difference;
  RealArray2D spline_gradient;
  RealArray3D spline_hessian;

  // local allocated
  RealArray initial_guess;
  RealArray2D d2jdu2;
  RealArray djdu;
  RealArray delta_guess;
  RealArray2D spline_gradient_AAt;
  RealArray2D search_bounds;
  RealArray lower_bound; // view to search bounds
  RealArray upper_bound; // view to search bounds
  IndexArray clipped;

  /// lhs matrix based on 2D array. Created by search methods
  std::unique_ptr<LhsMatrix> lhs_matrix;

  struct {
    double J; // .5 distance^2
    double distance;
    double squared_distance;
    double convergence_norm;
    int stop_iteration;
  } status;

  struct {
    int max_iter;
    double tolerance;
    // this should be set by calling ComputeConvergenceGoal()
    double convergence_goal;
  } options;

  /// @brief SLSQP specific search data. members are organized in the same order
  /// as the function call
  struct SLSQPData {
    /// n constraints, n equality constraints, max(m, 1), n variables
    int m, meq, la, n;

    /// current guess, lower bounds, upper bounds,
    double *x, *xl, *xu;

    /// value or objective (J)
    double f;

    /// m vector of constraints, partials of objective, m x n constraint normals
    double *c, *g, *a; // m vector of c constraints

    /// abs(acc) is accuracy
    double acc; // abs(acc) is accuracy

    /// max iteration and mode (see slsqp.c. 0 is for entry)
    int iter, mode;

    /// work spaces. SLSQPData::Setup() call should take care of this
    double* w; // work space -> see slsqp.f. need allocating
    int l_w;
    int* jw;
    int l_jw;

    /// slsqp internal variables
    double alpha, f0, gs, h1, h2, h3, h4, t, t0, tol;
    int iexact, incons, ireset, itermx, line, n1, n2, n3;

    /// @brief space
    RealArray w_space;
    IndexArray jw_space; // int array
    RealArray g_space;   // n + 1 array

    /// @brief proximity specific setup for SLSQP
    void Setup(SearchData& data);

    void CopyDerivative(const RealArray& der) { std::copy_n(der.data(), n, g); }
  } SLSQP;

  /// @brief Levenberg-Marquart specific search data
  struct LMData {
    double lambda;
    double prev_J;
    bool modified_update;
    double lower_bound;
    double upper_bound;
    RealArray metric;
    RealArray prev_guess;

    void Setup(SearchData& data) {
      metric.Reallocate(data.para_dim);
      prev_guess.Reallocate(data.para_dim);
    }

    /// @brief Computes || F(x) + J^T * dX ||^2
    /// @param spline_gradient_
    /// @param delta_guess_
    RealArray& Metric(const RealArray2D& spline_gradient_,
                      const RealArray& delta_guess_) {
      for (int i{}; i < metric.size(); ++i) {
        metric[i] = delta_guess_.InnerProduct(&spline_gradient_(i, 0));
      }
      return metric;
    }
  } LM;

  SearchData(const int para_dim,
             const int dim,
             const double tolerance,
             const int max_iter,
             const double* query,
             double* final_guess,
             double* nearest,
             double* nearest_minus_query,
             double* first_derivatives,
             double* second_derivatives) {
    Setup(para_dim,
          dim,
          tolerance,
          max_iter,
          query,
          final_guess,
          nearest,
          nearest_minus_query,
          first_derivatives,
          second_derivatives);
  }

  void Setup(const int para_dim_,
             const int dim_,
             const double tolerance,
             const int max_iter,
             const double* query,
             double* final_guess,
             double* nearest,
             double* nearest_minus_query,
             double* first_derivatives,
             double* second_derivatives);

  /// solvable lhs matrix - in mfem this is called Mult(rhs)
  LhsMatrix& Lhs() {
    if (!lhs_matrix) {
      splinepy::utils::PrintAndThrowError("lhs not set!");
    }
    return *lhs_matrix;
  }

  LhsMatrix& Lhs(RealArray2D& arr) {
    lhs_matrix = std::make_unique<LhsMatrix>(arr);
    return *lhs_matrix;
  }

  bool IsConverged() {
    return (status.convergence_norm < options.convergence_goal
            || status.distance < options.tolerance);
  }
};

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
  std::vector<int> bvh_samples_per_axis_;

  /// at most this many starting points are tried per query
  int max_starts_{4};
  /// at most this many elements are sampled per query
  int max_bvh_visits_{64};

  /// solvers are tried in this order, until one of them returns an acceptable
  /// result
  std::vector<Solver> solver_chain_{Solver::kDogleg,
                                    Solver::kSlsqp,
                                    Solver::kLevenbergMarquart};

  /// if true, a result is only accepted if it is not worse than the guess it
  /// started from, and if it is a (semi-)definite minimum within the bounds.
  bool guard_{true};

public:
  /// Constructor. As a spline helper class, always need a spline.
  explicit Proximity(splinepy::core::SplineView spline) : spline_(spline) {}

  /// @brief sets the order of solvers. Unknown/empty chains are not allowed.
  void SetSolverChain(const std::vector<Solver>& chain) {
    if (chain.empty()) {
      splinepy::utils::PrintAndThrowError("solver chain can't be empty");
    }
    solver_chain_ = chain;
  }

  void SetGuard(const bool guard) { guard_ = guard; }

  /// @brief Refreshes the view of the spline. The view refers to spans that
  /// the caller may have rebuilt (e.g. knot vector spans), so call this before
  /// each query session. Data (control points, ...) must be unchanged.
  void SetSpline(splinepy::core::SplineView spline) { spline_ = spline; }

  void SetMaxStarts(const int max_starts) {
    max_starts_ = std::max(1, max_starts);
  }

  /// A starting point for the local solvers.
  struct Start {
    /// squared distance from the query to the box of the element
    double lower_bound_squared;
    /// squared distance from the query to the sampled point
    double distance_squared;
    std::vector<double> guess;
    /// parametric distance between samples of the element, per axis
    std::vector<double> step;
  };

  /// @brief Builds a bounding volume hierarchy over the knot spans of the
  /// spline, as an alternative to the kd-tree. Elements are sampled lazily at
  /// query time. The total resolution is distributed to the elements, so that
  /// the number of samples is comparable to a kd-tree with the same
  /// resolutions. Returns false (and plants nothing) if the bounding boxes
  /// are not valid, e.g. due to non-positive weights.
  /// @param resolutions parameter space sampling resolution
  bool PlantBvh(const int* resolutions);

  bool HasBvh() const { return static_cast<bool>(bvh_); }

  /// @brief Candidate starting points of the local solvers, sorted by the
  /// distance of the sampled point. Contains one candidate per element that
  /// can hold a point as close as the closest sample.
  /// @param query
  std::vector<Start> BvhStarts(const double* query) const;

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
  void GuessMinusQuery(const RealArray& guess,
                       const ConstRealArray& query,
                       RealArray& difference) const;

  /// @brief difference = spline(guess) - query.
  /// This returns spline(guess)
  /// @param guess
  /// @param query
  /// @param difference
  /// @param guess_phys
  void GuessMinusQuery(const RealArray& guess,
                       const ConstRealArray& query,
                       RealArray& difference,
                       RealArray& guess_phys) const;

  /// @brief make initial guess from the initial guess source
  /// @param goal
  /// @param guess
  void MakeInitialGuess(const ConstRealArray& goal, RealArray& guess) const;

  /// @brief make initial guess from the initial guess source.
  /// Initial guess is saved to both initial_guess and current_guess
  /// @param data
  void MakeInitialGuess(SearchData& data) const;

  /// @brief updates current_guess with delta_guess and clips result
  /// @param data
  void UpdateAndClip(SearchData& data) const;

  /// @brief returns J = .5 * distance^2. Where distance is norm of difference.
  /// Difference is defined as || query - spline(guess) ||. Byproducts,
  /// difference and guess_phys (=spline(guess)) are also returned.  Used in
  /// ComputeCostAndDerivatives() and corresponds to depth=0.
  /// @param guess
  /// @param query
  /// @param difference
  /// @param guess_phys
  /// @return J
  double J(const RealArray& guess,
           const ConstRealArray& query,
           RealArray& difference,
           RealArray& guess_phys) const;

  /// @brief Fills dJ/du and a byproduct spline_gradient (dS/du). Used in
  /// ComputeCostAndDerivatives() and corresponds to depth=1.
  /// @param guess
  /// @param difference
  /// @param djdu
  /// @param spline_gradient
  void dJdu(const RealArray& guess,
            const RealArray& difference,
            RealArray& djdu,
            RealArray2D& spline_gradient) const;

  /// @brief Fills hessian of J (d2J/du2) and a byproduct spline_hessian
  /// (d2S/du2). Used in ComputeCostAndDerivatives() and corresponds to depth=2.
  /// @param guess
  /// @param difference
  /// @param spline_gradient_AAt
  /// @param d2jdu2
  /// @param spline_hessian
  void d2Jdu2(const RealArray& guess,
              const RealArray& difference,
              const RealArray2D& spline_gradient_AAt,
              RealArray2D& d2jdu2,
              RealArray3D& spline_hessian) const;

  /// @brief computes J and its derivatives based on the value of depth. depth
  /// == 1 -> jacobian, dept > 1 -> jacobian & hessian. Negative value is
  /// considered as highest depth we can compute (currently hessian)
  /// @param data
  /// @param depth
  void ComputeCostAndDerivatives(SearchData& data, int depth) const;

  /// @brief Computes status in SearchData based on current values in
  /// SearchData. If recompute_costs==true, it will compute the cost and its
  /// derivative again and store it.
  /// @param data
  /// @param recompute_costs
  void ComputeStatus(SearchData& data, const bool recompute_costs) const;

  /// @brief resets search bounds to spline's parametric bounds. if tight==true,
  /// current_guess +- the sampling step size of the initial guess source.
  /// @param data
  /// @param tight
  void FindSearchBound(SearchData& data, const bool tight = false) const;

  /// @brief Prepares lhs and rhs of newton iteration
  void PrepareIterationNewton(SearchData& data) const;

  /// @brief Performs newton iterations using PrepareIterationNewton
  /// @param data
  void Newton(SearchData& data) const;

  /// @brief Prepares lhs and rhs of LM iteration
  /// @param data
  void PrepareIterationLevenbergMarquart(SearchData& data) const;

  /// @brief Levenberg-Marquart method. Intended to be used as a fallback method
  /// for newton. Uses PrepareIterationLevenbergMarquart()
  /// @param data
  void LevenbergMarquart(SearchData& data) const;

  /// @brief norm of the gradient of J, after removing the components that
  /// point out of the search bounds. This vanishes at constrained minima.
  double ProjectedGradientNorm(const SearchData& data) const;

  /// @brief Trust region method with dogleg steps. Uses the exact hessian if it
  /// is positive definite, else the Gauss-Newton matrix. Respects search
  /// bounds by projecting steps and freezing variables that push out.
  /// @param data
  void Dogleg(SearchData& data) const;

  /// @brief is the current result acceptable? (see guard_). Fills the
  /// hessian of J and of the spline at the current guess as a byproduct.
  /// @param data
  /// @param reference_J J of the guess the solver started from
  bool IsAcceptable(SearchData& data, const double reference_J) const;

  /// @brief Prepares lhs and rhs of slsqp iteration and other parameters
  void PrepareIterationSlsqp(SearchData& data) const;

  /// @brief Performs slsqp iterations using PrepareIterationSlsqp
  /// @param data
  void Slsqp(SearchData& data) const;

  /// @brief Runs the solver chain from a given start. Same outputs as
  /// VerboseQuery().
  void SolveFrom(const double* query,
                 const double& tolerance,
                 const int& max_iterations,
                 const bool tight_bounds,
                 const double* start,
                 double* final_guess,
                 double* nearest,
                 double* nearest_minus_query,
                 double& distance,
                 double& convergence_norm,
                 double* first_derivatives,
                 double* second_derivatives) const;

  /// @brief Given physical coordinate, finds closest parametric coordinate.
  /// Takes initial guess from the initial guess source, or starts from several promising
  /// elements if a bvh was planted.
  ///
  /// @param[in] query
  /// @param[in] tolerance
  /// @param[in] max_iterations
  /// @param[in] tight_bounds
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
                    const bool tight_bounds,
                    double* final_guess,
                    double* nearest /* spline(final_guess) */,
                    double* nearest_minus_query /* difference */,
                    double& distance,
                    double& convergence_norm,
                    double* first_derivatives /* spline jacobian */,
                    double* second_derivatives /* spline hessian */) const;
};

/// @brief helper function to compute hessian at the end
/// @param spline
/// @param at
/// @param hess
void FillHessian(splinepy::core::SplineView spline,
                 const RealArray& at,
                 RealArray3D& hess);

} // namespace splinepy::proximity
