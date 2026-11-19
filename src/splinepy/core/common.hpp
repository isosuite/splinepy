#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

/// Marks functions that may also be compiled for a device (CUDA / HIP).
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SPLINEPY_HD __host__ __device__
#else
#define SPLINEPY_HD
#endif

/// Forces inlining, e.g. of the lambdas that unroll a loop over axes.
#if defined(__GNUC__) || defined(__clang__)
#define SPLINEPY_INLINE __attribute__((always_inline))
#else
#define SPLINEPY_INLINE
#endif

namespace splinepy::core {

constexpr double kDefaultTolerance = 1e-11;
constexpr std::size_t kDefaultAlignment = 64;

enum class Status : int32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kDimensionMismatch = 2,
  kOutOfBounds = 3,
  kNotConverged = 4,
  kDegenerate = 5,
};

inline std::string_view StatusToString(Status status) noexcept {
  switch (status) {
  case Status::kOk:
    return "Ok";
  case Status::kInvalidArgument:
    return "Invalid argument";
  case Status::kDimensionMismatch:
    return "Dimension mismatch";
  case Status::kOutOfBounds:
    return "Query out of bounds";
  case Status::kNotConverged:
    return "Numerical iteration failed to converge";
  case Status::kDegenerate:
    return "Degenerate geometry or zero Jacobian";
  default:
    return "Unknown error";
  }
}

class SplinepyException : public std::runtime_error {
public:
  explicit SplinepyException(const std::string& message,
                             Status status = Status::kInvalidArgument)
      : std::runtime_error(message),
        status_(status) {}

  explicit SplinepyException(const char* message,
                             Status status = Status::kInvalidArgument)
      : std::runtime_error(message),
        status_(status) {}

  Status status() const noexcept { return status_; }

private:
  Status status_;
};

} // namespace splinepy::core
