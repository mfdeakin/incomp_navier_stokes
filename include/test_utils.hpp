
#include <cmath>
#include <functional>
#include <utility>

namespace TestUtils {

// Attributes don't seem to be supported by clang-format yet... See
// https://stackoverflow.com/questions/45740466/clang-format-setting-to-control-c-attributes
// clang-format off

[[nodiscard]] real error(const real expected, const real computed) {
  return std::abs(expected - computed);
}

// Computes the order and extrapolated value given three estimated values
// for discretizations which double the granularity at each step
[[nodiscard]] std::pair<real, real> richardson(const real fine,
                                               const real medium,
                                               const real coarse) {
  const double order_r = (fine - medium) / (medium - coarse);
  const double order   = -std::log2(order_r);

  const double extrap = fine - (medium - fine) * (1 / (std::pow(2, order) - 1));
  return {order, extrap};
}
// clang-format on

// Computes the error of the computed value relative to the correct value
// This tells us how many bits are wring in the mantissa
[[nodiscard]] real rel_error(const real expected, const real computed) {
  if(expected == 0.0) {
    return computed;
  }
  return std::abs((expected - computed) / expected);
}

template <typename MeshT>
void fill_mesh(MeshT &mesh, const std::function<real(real, real)> &val) {
  for(int i = 0; i < mesh.extent(0); i++) {
    for(int j = 0; j < mesh.extent(1); j++) {
      const real x = mesh.x_median(i);
      const real y = mesh.y_median(j);

      mesh(i, j) = val(x, y);
    }
  }
}

}  // namespace TestUtils
