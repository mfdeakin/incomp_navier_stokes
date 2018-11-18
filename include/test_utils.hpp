
#include <cmath>
#include <functional>
#include <numeric>
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
[[nodiscard]] real linf_error(MeshT &mesh,
                            std::function<real(const MeshT &, int, int)> err) {
  real reduced = 0.0;
  for(auto itr = mesh.begin(); itr != mesh.end(); ++itr) {
    const int i = itr.index(0);
    const int j = itr.index(1);

    const real diff = err(mesh, i, j);

    reduced = std::max(reduced, std::abs(diff));
  }
  return reduced;
}

template <typename MeshT>
[[nodiscard]] real l1_error(MeshT &mesh,
                            std::function<real(const MeshT &, int, int)> err) {
  real reduced = 0.0;
  for(auto itr = mesh.begin(); itr != mesh.end(); ++itr) {
    const int i = itr.index(0);
    const int j = itr.index(1);

    const real diff = err(mesh, i, j);
    reduced += std::abs(diff);
  }
  return reduced;
}

template <typename MeshT>
[[nodiscard]] real l2_error(MeshT &mesh,
                            std::function<real(const MeshT &, int, int)> err) {
  real reduced = 0.0;
  for(auto itr = mesh.begin(); itr != mesh.end(); ++itr) {
    const int i = itr.index(0);
    const int j = itr.index(1);

    const real diff = err(mesh, i, j);
    reduced += (diff * diff);
  }
  return reduced;
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
