
#include <utility>

namespace TestUtils {

// Computes the order and extrapolated value given three estimated values for
// discretizations which double the granularity at each step
[[nodiscard]] std::pair<real, real> richardson(const real fine, const real medium,
                                 const real coarse) {
  const double order_r = (fine - medium) / (medium - coarse);
  const double order   = -std::log2(order_r);

  const double extrap = fine - (medium - fine) * (1 / (std::pow(2, order) - 1));
  return {order, extrap};
}

}  // namespace TestUtils
