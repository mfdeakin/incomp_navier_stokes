
#ifndef _CONSTANTS_HPP_
#define _CONSTANTS_HPP_

#include <limits>

using real = double;

constexpr real pi = 3.1415926535897932384626433832795;

constexpr real prandtl  = 0.7;
constexpr real eckert   = 0.1;

constexpr real q_nan = std::numeric_limits<real>::quiet_NaN();
constexpr real s_nan = std::numeric_limits<real>::signaling_NaN();

#endif  // _CONSTANTS_HPP_
