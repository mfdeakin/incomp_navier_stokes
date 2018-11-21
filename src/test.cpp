
#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "test_utils.hpp"
#include "thomas.hpp"

using rngAlg = std::mt19937_64;

// TEST(utility, fill_mesh) {
//   constexpr int ctrl_vols_x = 128, ctrl_vols_y = 128;
//   constexpr real min_x = 0.0, min_y = 0.0;
//   constexpr real max_x = 0.1, max_y = 0.1;
//   using MeshT = Mesh<ctrl_vols_x, ctrl_vols_y>;
//   std::unique_ptr<MeshT> mesh =
//       std::make_unique<MeshT>(min_x, max_x, min_y, max_y);
//   EnergyAssembly<SecondOrderCentered_Part1> space_disc(1.0, pi, 1.0 / pi);
//   TestUtils::fill_mesh(*mesh, space_disc.solution_tuple());

//   for(int i = 0; i < mesh->x_dim(); i++) {
//     for(int j = 0; j < mesh->y_dim(); j++) {
//       const real x = mesh->x_median(i);
//       const real y = mesh->y_median(j);
//       EXPECT_EQ(mesh->Temp(i, j), space_disc.solution(x, y));
//       EXPECT_EQ(mesh->u_vel(i, j), space_disc.u(x, y));
//       EXPECT_EQ(mesh->v_vel(i, j), space_disc.v(x, y));
//     }
//   }
// }

// // Verifies that the boundary flux calculations for x, y, dx, and dy are
// correct
// // Uses the isotropic test function T = x * sin(pi * x) + y * sin(pi * y)
// // with a constant velocity field
// TEST(part_1, fluxes_calc_1) {
//   constexpr real max_err    = 1e-6;
//   constexpr int ctrl_vols_x = 128, ctrl_vols_y = 128;
//   constexpr real min_x = 0.0, min_y = 0.0;
//   constexpr real max_x = 0.1, max_y = 0.1;
//   using MeshT = Mesh<ctrl_vols_x, ctrl_vols_y>;
//   std::unique_ptr<MeshT> mesh =
//       std::make_unique<MeshT>(min_x, max_x, min_y, max_y);
//   EnergyAssembly<SecondOrderCentered_Part1> space_disc(q_nan, q_nan, q_nan);

//   auto sol = [](const real x, const real y) {
//     return std::tuple<real, real, real>(
//         (x * std::sin(pi * x)) + (y * std::sin(pi * y)), 1.0, 1.0);
//   };

//   auto sol_dx = [](const real x, const real y) {
//     return (x * pi * std::cos(pi * x) + std::sin(pi * x));
//   };

//   auto sol_dy = [=](const real x, const real y) { return sol_dx(y, x); };

//   TestUtils::fill_mesh(*mesh, sol);

//   // The flux computation assumes the boundary value,
//   // so skip the last value which can't be computed from the mesh
//   for(int i = 0; i < ctrl_vols_x - 1; i++) {
//     for(int j = 0; j < ctrl_vols_y - 1; j++) {
//       const real x = mesh->x_median(i);
//       const real y = mesh->y_median(j);

//       const real x_flux = std::get<0>(sol(mesh->x_max(i), y));
//       const real y_flux = std::get<0>(sol(x, mesh->y_max(j)));

//       EXPECT_NEAR(x_flux, space_disc.uT_x_flux(*mesh, i, j), max_err);
//       EXPECT_NEAR(y_flux, space_disc.vT_y_flux(*mesh, i, j), max_err);

//       const real dx_flux = sol_dx(mesh->x_max(i), y);
//       const real dy_flux = sol_dy(x, mesh->y_max(j));

//       EXPECT_NEAR(dx_flux, space_disc.dx_flux(*mesh, i, j), max_err);
//       EXPECT_NEAR(dy_flux, space_disc.dy_flux(*mesh, i, j), max_err);
//     }
//   }
// }

// // Verifies that the boundary flux calculations for x, y, dx, and dy are
// correct
// // Uses the solution for part 1
// TEST(part_1, fluxes_calc_2) {
//   constexpr real max_err    = 1e-6;
//   constexpr int ctrl_vols_x = 2500, ctrl_vols_y = 2500;
//   constexpr real min_x = 0.0, min_y = 0.0;
//   constexpr real max_x = 1.0, max_y = 1.0;
//   using MeshT = Mesh<ctrl_vols_x, ctrl_vols_y>;
//   std::unique_ptr<MeshT> mesh =
//       std::make_unique<MeshT>(min_x, max_x, min_y, max_y);
//   EnergyAssembly<SecondOrderCentered_Part1> space_disc(1.0, pi, 1.0 / pi);

//   TestUtils::fill_mesh(*mesh, space_disc.solution_tuple());

//   // Start at negative 1 to ensure the boundaries are implemented correctly
//   for(int i = -1; i < ctrl_vols_x; i++) {
//     for(int j = -1; j < ctrl_vols_y; j++) {
//       const real x = mesh->x_median(i);
//       const real y = mesh->y_median(j);

//       // There isn't a well defined value when the coordinate of the
//       dimension
//       // the flux isn't being computed for is outside of the mesh
//       if(j != -1) {
//         const real x_max = mesh->x_max(i);
//         EXPECT_NEAR(space_disc.solution(x_max, y) * space_disc.u(x_max, y),
//                     space_disc.uT_x_flux(*mesh, i, j), max_err);
//         EXPECT_NEAR(space_disc.solution_dx(x_max, y),
//                     space_disc.dx_flux(*mesh, i, j), max_err);
//         // Sanity check for the derivative solution implementation
//         EXPECT_NEAR((space_disc.solution(x + mesh->dx(), y) -
//                      space_disc.solution(x, y)) /
//                         mesh->dx(),
//                     space_disc.dx_flux(*mesh, i, j), max_err);
//       }
//       if(i != -1) {
//         const real y_max = mesh->y_max(j);
//         EXPECT_NEAR(space_disc.solution(x, y_max) * space_disc.v(x, y_max),
//                     space_disc.vT_y_flux(*mesh, i, j), max_err);
//         EXPECT_NEAR(space_disc.solution_dy(x, y_max),
//                     space_disc.dy_flux(*mesh, i, j), max_err);
//         // Sanity check for the derivative solution implementation
//         EXPECT_NEAR((space_disc.solution(x, y + mesh->dy()) -
//                      space_disc.solution(x, y)) /
//                         mesh->dy(),
//                     space_disc.dy_flux(*mesh, i, j), max_err);
//       }
//     }
//   }
// }

// // Use the same function as in fluxes_calc_1 to verify the flux integral is
// // correct
// TEST(part_1, flux_integral_2) {
//   constexpr int ctrl_vols_x = 2560, ctrl_vols_y = 2560;

//   using MeshT = Mesh<ctrl_vols_x, ctrl_vols_y>;
//   EnergyAssembly<SecondOrderCentered_Part1> space_disc(1.0, 1.0, 1.0);
//   std::unique_ptr<MeshT> mesh =
//       std::make_unique<MeshT>(space_disc.x_min(), space_disc.x_max(),
//                               space_disc.y_min(), space_disc.y_max());

//   TestUtils::fill_mesh(*mesh, [=](const real x, const real y) {
//     return std::tuple<real, real, real>(space_disc.solution(x, y), 1.0, 1.0);
//   });

//   for(int i = 0; i < ctrl_vols_x - 1; i++) {
//     // On the y boundaries, dT/dy has catastrophic cancellation problems,
//     // so ignore them and just trust our flux calculation tests
//     for(int j = 1; j < ctrl_vols_y - 2; j++) {
//       const real x     = mesh->x_median(i);
//       const real y     = mesh->y_median(j);
//       const real x_min = mesh->x_min(i);
//       const real y_min = mesh->y_min(j);
//       const real x_max = mesh->x_max(i);
//       const real y_max = mesh->y_max(j);

//       const real d2t_dx2 = (space_disc.solution_dx(x_max, y) -
//                             space_disc.solution_dx(x_min, y)) /
//                            mesh->dx();

//       EXPECT_NEAR(d2t_dx2, space_disc.solution_dx2(x, y), 5e-6);

//       const real x2_deriv = (space_disc.dx_flux(*mesh, i, j) -
//                              space_disc.dx_flux(*mesh, i - 1, j)) /
//                             mesh->dx();

//       EXPECT_NEAR(d2t_dx2, x2_deriv, 1e-6);

//       const real y2_deriv = (space_disc.dy_flux(*mesh, i, j) -
//                              space_disc.dy_flux(*mesh, i, j - 1)) /
//                             mesh->dy();

//       const real d2t_dy2 = (space_disc.solution_dy(x, y_max) -
//                             space_disc.solution_dy(x, y_min)) /
//                            mesh->dy();

//       EXPECT_NEAR(d2t_dy2, y2_deriv, 1e-6);

//       // const real diffuse = (d2t_dx2 + d2t_dy2) / (reynolds * prandtl);

//       // EXPECT_EQ((x2_deriv + y2_deriv) / (reynolds * prandtl),
//       //           space_disc.flux_integral(*mesh, i, j));

//       // printf(
//       //     "%2d %2d (% .3e, % .3e): "
//       //     "% .6e vs % .6e vs % .6e\n",
//       //     i, j, x, y, space_disc_diffuse.solution_dy2(x, y),
//       //     space_disc_diffuse.flux_integral(*mesh, i, j),
//       //     2.0 * pi * pi * cos(pi * x) * sin(pi * y));

//       // EXPECT_NEAR(diffuse, space_disc.flux_integral(*mesh, i, j), 5e-5);

//       // const real dt_dx =
//       //     (space_disc_u.solution(x_max, y) * space_disc_u.u(x_max, y) -
//       //      space_disc_u.solution(x_min, y) * space_disc_u.u(x_min, y)) /
//       //     mesh->dx();
//       // EXPECT_NEAR(dt_dx, space_disc.solution_dx(x, y) * space_disc.u(x,
//       y),
//       // 1e-5);

//       // EXPECT_NEAR(dt_dx, -space_disc_u.flux_integral(*mesh, i, j), 1e-6);

//       // const real dt_dy = space_disc_v.solution_dy(x, y) *
//       space_disc_v.v(x,
//       // y); EXPECT_NEAR(dt_dy, -space_disc_v.flux_integral(*mesh, i, j),
//       // 1e-6);

//       // EXPECT_NEAR(space_disc.flux_integral(*mesh, i, j),
//       //             -dt_dx - dt_dy + (d2t_dx2 + d2t_dy2) / (reynolds *
//       //             prandtl), 5e-5);
//     }
//   }
// }

constexpr int max_mesh_size = 500;

// This computes the L1 error and the order of minimum convergence of the
// solution for each cell in the fine mesh.
// This requires that you specify how the error is computed with the est_error
// functor, and how the order of convergence of the cell values are computed
// with the est_order functor
template <typename FineMesh, typename MedMesh, typename CoarseMesh>
[[nodiscard]] std::tuple<int, int, int, real, real> compute_error_min_conv(
    FineMesh &fine, const MedMesh &med, const CoarseMesh &coarse,
    std::function<real(const FineMesh &, const MedMesh &, const CoarseMesh &,
                       const int, const int)>
        est_order,
    std::function<real(const FineMesh &, const int, const int)>
        est_error) noexcept {
  real min_order = std::numeric_limits<real>::infinity();
  int order_i = -1, order_j = -1;
  for(int i = 1; i < FineMesh::x_dim() - 1; i++) {
    for(int j = 1; j < FineMesh::y_dim() - 1; j++) {
      const real order = est_order(fine, med, coarse, i, j);
      if(order < min_order) {
        min_order = order;
        order_i   = i;
        order_j   = j;
      }
    }
  }
  const real l1_err = TestUtils::l1_error(fine, est_error);

  return {FineMesh::x_dim(), order_i, order_j, min_order, l1_err};
}

// This function recursively calls the function with twice it's mesh size until
// the maximum mesh size becomes too large.
// This is done by conditionally enabling (with SFINAE) the computation of
// properties about these meshes based on their size.
// Each function returns the mesh it created and the next largest mesh;
// this makes computation of the mesh order easier
// To limit the runtime memory required, we only keep three meshes in scope at
// any given time. This requires that we return meshes rather than pass them up,
// making this code much more complicated (sorry)
template <typename MeshT>
typename std::enable_if<
    MeshT::x_dim() * 2 >= max_mesh_size,
    // Return the current mesh and the next largest mesh
    std::pair<std::unique_ptr<MeshT>,
              std::unique_ptr<Mesh<MeshT::x_dim() * 2, MeshT::y_dim() * 2>>>>::
    // yuck, too many >>>>>>>>>>>>>>>>>...
    type
    compute_mesh_errors(const SecondOrderCentered_Part1 &space_disc,
                        std::vector<std::tuple<int, real, real>> &vec) {
  using MedMesh  = MeshT;
  using FineMesh = Mesh<MedMesh::x_dim() * 2, MedMesh::y_dim() * 2>;
  auto med_mesh =
      std::make_unique<MedMesh>(space_disc.x_min(), space_disc.x_max(),
                                space_disc.y_min(), space_disc.y_max());
  // Ensure it's initialized with the solution
  TestUtils::fill_mesh(*med_mesh, space_disc.solution_tuple());

  auto fine_mesh =
      std::make_unique<FineMesh>(space_disc.x_min(), space_disc.x_max(),
                                 space_disc.y_min(), space_disc.y_max());
  TestUtils::fill_mesh(*fine_mesh, space_disc.solution_tuple());
  return std::pair<std::unique_ptr<MedMesh>, std::unique_ptr<FineMesh>>(
      std::move(med_mesh), std::move(fine_mesh));
}

template <typename MeshT>
std::tuple<std::unique_ptr<MeshT>,
           std::unique_ptr<Mesh<MeshT::x_dim() * 2, MeshT::y_dim() * 2>>,
           std::unique_ptr<Mesh<MeshT::x_dim() * 4, MeshT::y_dim() * 4>>>
compute_mesh_errs_init(const SecondOrderCentered_Part1 &space_disc,
                       std::vector<std::tuple<int, real, real>> &vec) {
  constexpr int x_dim = MeshT::x_dim();
  constexpr int y_dim = MeshT::y_dim();
  using MedMesh       = Mesh<x_dim * 2, y_dim * 2>;
  // Before we create our own mesh, we need to get the next two
  // This ensures that no more than 3 meshes are allocated at any given time
  auto [med_mesh, fine_mesh] = compute_mesh_errors<MedMesh>(space_disc, vec);

  // Now just initialize the coarse mesh and compute the results for the fine
  // mesh
  auto coarse_mesh =
      std::make_unique<MeshT>(space_disc.x_min(), space_disc.x_max(),
                              space_disc.y_min(), space_disc.y_max());
  TestUtils::fill_mesh(*coarse_mesh, space_disc.solution_tuple());
  return {std::move(coarse_mesh), std::move(med_mesh), std::move(fine_mesh)};
}

// The actual implementation does more than just initialize meshes :)
template <typename MeshT>
typename std::enable_if<
    MeshT::x_dim() * 2 < max_mesh_size,
    std::pair<std::unique_ptr<MeshT>,
              std::unique_ptr<Mesh<MeshT::x_dim() * 2, MeshT::y_dim() * 2>>>>::
    type
    compute_mesh_errors(const SecondOrderCentered_Part1 &space_disc,
                        std::vector<std::tuple<int, real, real>> &vec) {
  auto [coarse_mesh, med_mesh, fine_mesh] =
      compute_mesh_errs_init<MeshT>(space_disc, vec);

  constexpr int x_dim = MeshT::x_dim();
  constexpr int y_dim = MeshT::y_dim();

  using CoarseMesh = MeshT;
  using MedMesh    = Mesh<x_dim * 2, y_dim * 2>;
  using FineMesh   = Mesh<x_dim * 4, y_dim * 4>;

  std::function<real(const FineMesh &, const MedMesh &, const CoarseMesh &,
                     const int, const int)>
      est_order = [=](const FineMesh &fine, const MedMesh &med,
                      const MeshT &coarse, const int i, const int j) {
        const real x                    = fine.x_median(i);
        const real y                    = fine.y_median(j);
        const auto [med_i, med_j]       = med.cell_idx(x, y);
        const auto [coarse_i, coarse_j] = coarse.cell_idx(x, y);

        const real fine_val = space_disc.source_fd(fine, i, j);
        const real med_val  = space_disc.source_fd(med, med_i, med_j);
        const real coarse_val =
            space_disc.source_fd(coarse, coarse_i, coarse_j);

        const auto [order, extrap] =
            TestUtils::richardson(fine_val, med_val, coarse_val);

        if(std::abs(x - 0.375) < 0.002 && std::abs(y - 0.125) < 0.002) {
          printf(
              "%4d for (% .4e, % .4e) (% .4e, % .4e) (% .4e, % .4e): "
              "% .9e, % .9e, % .9e => % .9e vs % .9e, % .9e\n",
              FineMesh::x_dim(), coarse.x_median(coarse_i),
              coarse.y_median(coarse_j), med.x_median(med_i),
              med.y_median(med_j), x, y,
              coarse_val - space_disc.source_sol(x, y),
              med_val - space_disc.source_sol(x, y),
              fine_val - space_disc.source_sol(x, y), order,
              std::sqrt((coarse_val - space_disc.source_sol(x, y)) /
                        (med_val - space_disc.source_sol(x, y))),
              std::sqrt((med_val - space_disc.source_sol(x, y)) /
                        (fine_val - space_disc.source_sol(x, y))));
        }

        return order;
      };

  std::function<real(const FineMesh &, const int i, const int j)> est_err =
      [=](const FineMesh &mesh, const int i, const int j) {
        const real x = mesh.x_median(i);
        const real y = mesh.y_median(j);
        return space_disc.solution(x, y) - mesh.Temp(i, j);
      };

  std::tuple<int, int, int, real, real> t =
      compute_error_min_conv<FineMesh, MedMesh, CoarseMesh>(
          *fine_mesh, *med_mesh, *coarse_mesh, est_order, est_err);

  const int i                     = std::get<1>(t);
  const int j                     = std::get<2>(t);
  const real x                    = fine_mesh->x_median(i);
  const real y                    = fine_mesh->y_median(j);
  const auto [med_i, med_j]       = med_mesh->cell_idx(x, y);
  const auto [coarse_i, coarse_j] = coarse_mesh->cell_idx(x, y);

  printf(
      "%d %d Minimum order of convergence at %3d %3d (%.3e, %.3e): "
      "% .8e, % .8e, % .8e => % .8e\n",
      std::get<0>(t), x_dim * 4, i, j, x, y, fine_mesh->Temp(i, j),
      med_mesh->Temp(med_i, med_j), coarse_mesh->Temp(coarse_i, coarse_j),
      std::get<3>(t));

  vec.push_back({std::get<0>(t), std::get<3>(t), std::get<4>(t)});
  return std::pair<std::unique_ptr<CoarseMesh>, std::unique_ptr<MedMesh>>{
      std::move(coarse_mesh), std::move(med_mesh)};
}

// Verify the accuracy of the source term is second order with mesh size
TEST(part_2, source_term) {
  constexpr int ctrl_vols_x = 640, ctrl_vols_y = 640;

  using MeshT = Mesh<ctrl_vols_x, ctrl_vols_y>;
  EnergyAssembly<SecondOrderCentered_Part1> space_disc(1.0, 1.0, 1.0);
  std::unique_ptr<MeshT> mesh =
      std::make_unique<MeshT>(space_disc.x_min(), space_disc.x_max(),
                              space_disc.y_min(), space_disc.y_max());

  TestUtils::fill_mesh(*mesh, space_disc.solution_tuple());

  for(int i = 0; i < mesh->x_dim(); i++) {
    for(int j = 0; j < mesh->y_dim(); j++) {
      const real x = mesh->x_median(i);
      const real y = mesh->y_median(j);
      EXPECT_NEAR(space_disc.source_sol(x, y),
                  space_disc.source_fd(*mesh, i, j), 1e-6);
    }
  }
}

TEST(part_1, flux_integral_convergence) {
  using err_tuple = std::tuple<int, real, real>;
  SecondOrderCentered_Part1 space_disc(1.0, 1.0, 1.0);
  std::vector<err_tuple> errors;
  compute_mesh_errors<Mesh<10, 10>>(space_disc, errors);
  // For the solution to be correct, the order of convergence (as the mesh
  // grows) must match the order of the spatial discretization, and the
  // extrapolated estimate should go to 0
  for(const auto [m_dim, l1_err, min_order] : errors) {
    EXPECT_NEAR(2.0, min_order, 2e-1);
    printf(
        "Richardson Minimum Estimated Order: % .3e; L1 Error: % .3e;"
        " for mesh %d\n",
        min_order, l1_err, m_dim);
  }
}

TEST(thomas, thomas) {
  constexpr int ctrl_vols = 10;
  ND_Array<real, ctrl_vols + 2, 3> mtx_diags;
  ND_Array<real, ctrl_vols + 2> rhs;
  for(int i = 0; i < mtx_diags.extent(0); i++) {
    mtx_diags(i, 0) = 1.0;
    mtx_diags(i, 1) = 2.0 + i;
    mtx_diags(i, 2) = 3.0;
    rhs(i)          = i;
  }
  SolveThomas(mtx_diags, rhs);
  for(int i = 0; i < ctrl_vols; i++) {
    // We stored the result in rhs, so applying the matrix operation to it
    // should get us back to the original rhs
    double result;
    if(i == 0) {
      result = rhs(i) * 2.0 + rhs(i + 1) * 3.0;
    } else if(i == 11) {
      result = rhs(i - 1) * 1.0 + rhs(i) * (2.0 + i);
    } else {
      result = rhs(i - 1) * 1.0 + rhs(i) * (2.0 + i) + rhs(i + 1) * 3.0;
    }
    EXPECT_NEAR(i, result, 1e-10);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
