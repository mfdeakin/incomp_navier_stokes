
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
[[nodiscard]] real linf_error(
    MeshT &mesh, std::function<real(const MeshT &, int, int)> err) {
  real reduced = 0.0;
  for(auto itr = mesh.Temp().begin(); itr != mesh.Temp().end(); ++itr) {
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
  for(auto itr = mesh.Temp().begin(); itr != mesh.Temp().end(); ++itr) {
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
  for(auto itr = mesh.Temp().begin(); itr != mesh.Temp().end(); ++itr) {
    const int i = itr.index(0);
    const int j = itr.index(1);

    const real diff = err(mesh, i, j);
    reduced += (diff * diff);
  }
  return reduced;
}

template <typename MeshT>
void fill_mesh(
    MeshT &mesh,
    const std::function<std::tuple<real, real, real>(real, real)> &val) {
  for(int i = 0; i < mesh.x_dim(); i++) {
    for(int j = 0; j < mesh.y_dim(); j++) {
      const real x = mesh.x_median(i);
      const real y = mesh.y_median(j);

      const auto [T, u, v] = val(x, y);

      mesh.Temp(i, j)  = T;
      mesh.u_vel(i, j) = u;
      mesh.v_vel(i, j) = v;
    }
  }
}

// This computes the L1 error and the order of minimum convergence of the
// solution for each cell in the fine mesh.
// This requires that you specify how the error is computed with the est_error
// functor, and how the order of convergence of the cell values are computed
// with the est_order functor
template <typename FineMesh, typename CoarseMesh>
[[nodiscard]] std::tuple<int, int, int, real, real> compute_error_min_conv(
    FineMesh &fine, const CoarseMesh &coarse,
    std::function<real(const FineMesh &, const int, const int)> est_error_fine,
    std::function<real(const CoarseMesh &, const int, const int)>
        est_error_coarse) noexcept {
  real min_order = std::numeric_limits<real>::infinity();
  int order_i = -1, order_j = -1;
  for(int i = 1; i < CoarseMesh::x_dim() - 1; i++) {
    for(int j = 1; j < CoarseMesh::y_dim() - 1; j++) {
      const real coarse_err = est_error_coarse(coarse, i, j);
      const auto [fine_i, fine_j] =
          fine.cell_idx(coarse.x_median(i), coarse.y_median(j));
      const real fine_err = est_error_fine(fine, fine_i, fine_j);
      if(fine_err != 0.0) {
        const real order = std::log2(coarse_err / fine_err);
        if(order < min_order) {
          min_order = order;
          order_i   = i;
          order_j   = j;
        }
      }    }
  }
  const real l1_err = TestUtils::l1_error(fine, est_error_fine);

  return {FineMesh::x_dim(), order_i, order_j, min_order, l1_err};
}

template <typename MeshT>
std::pair<std::unique_ptr<MeshT>,
          std::unique_ptr<Mesh<MeshT::x_dim() * 2, MeshT::y_dim() * 2>>>
compute_mesh_errs_init(
    const SecondOrderCentered_Part1 &space_disc,
    std::vector<std::tuple<int, real, real>> &vec,
    std::function<std::unique_ptr<Mesh<MeshT::x_dim() * 2, MeshT::y_dim() * 2>>(
        const SecondOrderCentered_Part1 &space_disc,
        std::vector<std::tuple<int, real, real>> &vec)>
        compute_errs) {
  constexpr int x_dim = MeshT::x_dim();
  constexpr int y_dim = MeshT::y_dim();
  using FineMesh      = Mesh<x_dim * 2, y_dim * 2>;
  // Before we create our own mesh, we need to get the next two
  // This ensures that no more than 3 meshes are allocated at any given time
  std::unique_ptr<FineMesh> fine_mesh = compute_errs(space_disc, vec);

  // Now just initialize the coarse mesh and compute the results for the fine
  // mesh
  std::unique_ptr<MeshT> coarse_mesh =
      std::make_unique<MeshT>(space_disc.x_min(), space_disc.x_max(),
                              space_disc.y_min(), space_disc.y_max());
  TestUtils::fill_mesh(*coarse_mesh, space_disc.solution_tuple());
  return {std::move(coarse_mesh), std::move(fine_mesh)};
}

}  // namespace TestUtils
