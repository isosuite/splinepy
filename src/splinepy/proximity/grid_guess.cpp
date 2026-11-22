#include "splinepy/proximity/grid_guess.hpp"

#include <cstdint>
#include <limits>

#include "splinepy/core/math/uniform_grid.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::proximity {
namespace {

class GridGuess final : public InitialGuess {
public:
  GridGuess(splinepy::core::SplineView spline,
            const int* resolutions,
            const int n_thread)
      : dim_(spline.dim) {
    splinepy::core::ThrowIfInvalid(spline);
    const int32_t para_dim = spline.para_dim;

    const auto bounds = splinepy::core::ParametricBounds(spline);
    grid_ =
        splinepy::core::UniformGrid(para_dim,
                                    bounds.cspan(),
                                    splinepy::core::Span<const int32_t>(
                                        resolutions,
                                        static_cast<std::size_t>(para_dim)));

    // the spline was validated above; one batched evaluation of the grid
    splinepy::core::Options options;
    options.validate = false;
    options.nthreads = n_thread;
    const auto points = grid_.Points();
    samples_ = splinepy::core::Evaluate(spline, points.cspan(), options);
  }

  void Guess(const double* query, double* guess) const override {
    const int64_t n_samples = grid_.Size();
    int64_t best_id{};
    double best = std::numeric_limits<double>::max();
    for (int64_t i{}; i < n_samples; ++i) {
      const double* x = samples_.data() + i * dim_;
      double d2{};
      for (int k{}; k < dim_; ++k) {
        d2 += (x[k] - query[k]) * (x[k] - query[k]);
      }
      if (d2 < best) {
        best = d2;
        best_id = i;
      }
    }
    grid_.Point(best_id, guess);
  }

  const double* StepSize() const override { return grid_.StepSizes().data(); }

private:
  int dim_;
  splinepy::core::UniformGrid grid_;
  splinepy::core::Buffer<double> samples_;
};

} // namespace

std::unique_ptr<InitialGuess> MakeGridGuess(splinepy::core::SplineView spline,
                                            const int* resolutions,
                                            const int n_thread) {
  return std::make_unique<GridGuess>(spline, resolutions, n_thread);
}

} // namespace splinepy::proximity
