
#ifndef _CONSTANTS_HPP_
#define _CONSTANTS_HPP_

#include <limits>

using real = double;

constexpr real pi = 3.1415926535897932;

constexpr real reynolds = 50.0;
constexpr real prandtl  = 0.7;
constexpr real eckert   = 0.1;

constexpr real q_nan = std::numeric_limits<real>::quiet_NaN();

#endif  // _CONSTANTS_HPP_
