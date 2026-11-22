#pragma once

#include <sstream>

#include "splinepy/core/common.hpp"

namespace splinepy::core {

/// Throws a SplinepyException whose message is "SPLINEPY ERROR - " followed
/// by every argument and a space, then a newline. This is the message format
/// splinepy always used for these errors, so code matching on it keeps
/// working.
template<typename... Args>
[[noreturn]] void ThrowError(Status status, const Args&... args) {
  std::ostringstream message;
  message << "SPLINEPY ERROR - ";
  ((message << args << " "), ...);
  message << "\n";
  throw SplinepyException(message.str(), status);
}

} // namespace splinepy::core
