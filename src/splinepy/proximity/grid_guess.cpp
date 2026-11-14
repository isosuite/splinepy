#include "splinepy/proximity/grid_guess.hpp"

#include <limits>
#include <vector>

#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"
#include "splinepy/utils/grid_points.hpp"
#include "splinepy/utils/nthreads.hpp"

namespace splinepy::proximity {
namespace {

/// the spline is validated once, on construction
const splinepy::core::Options kNoValidation = []() {
  splinepy::core::Options options;
  options.validate = false;
  return options;
}();

class GridGuess final : public InitialGuess {
public:
  GridGuess(splinepy::core::SplineView spline,
            const int* resolutions,
            const int n_thread)
      : dim_(spline.dim) {
    splinepy::core::ThrowIfInvalid(spline);
    const int para_dim = spline.para_dim;

    const auto bounds = splinepy::core::ParametricBounds(spline);
    grid_points_.SetUp(para_dim, bounds.data(), resolutions);

    const int n_samples = grid_points_.Size();
    samples_.resize(static_cast<std::size_t>(n_samples) * dim_);

    auto sample = [&](const int begin, const int end, int) {
      std::vector<double> u(para_dim);
      for (int i{begin}; i < end; ++i) {
        grid_points_.IdToGridPoint(i, u.data());
        splinepy::core::Evaluate(
            spline,
            splinepy::core::Span<const double>(u.data(), para_dim),
            splinepy::core::Span<double>(&samples_[i * dim_], dim_),
            kNoValidation);
      }
    };
    splinepy::utils::NThreadExecution(sample, n_samples, n_thread);
  }

  void Guess(const double* query, double* guess) const override {
    const int n_samples = grid_points_.Size();
    int best_id{};
    double best = std::numeric_limits<double>::max();
    for (int i{}; i < n_samples; ++i) {
      const double* x = &samples_[static_cast<std::size_t>(i) * dim_];
      double d2{};
      for (int k{}; k < dim_; ++k) {
        d2 += (x[k] - query[k]) * (x[k] - query[k]);
      }
      if (d2 < best) {
        best = d2;
        best_id = i;
      }
    }
    grid_points_.IdToGridPoint(best_id, guess);
  }

  const double* StepSize() const override {
    return grid_points_.step_size_.data();
  }

private:
  int dim_;
  splinepy::utils::GridPoints grid_points_;
  std::vector<double> samples_;
};

} // namespace

std::unique_ptr<InitialGuess> MakeGridGuess(splinepy::core::SplineView spline,
                                            const int* resolutions,
                                            const int n_thread) {
  return std::make_unique<GridGuess>(spline, resolutions, n_thread);
}

} // namespace splinepy::proximity
