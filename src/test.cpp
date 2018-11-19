
#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "test_utils.hpp"

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
//   constexpr int ctrl_vols_x = 2560, ctrl_vols_y = 2560;
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
//       }
//       if(i != -1) {
//         const real y_max = mesh->y_max(j);
//         EXPECT_NEAR(space_disc.solution(x, y_max) * space_disc.v(x, y_max),
//                     space_disc.vT_y_flux(*mesh, i, j), max_err);
//         EXPECT_NEAR(space_disc.solution_dy(x, y_max),
//                     space_disc.dy_flux(*mesh, i, j), max_err);
//       }
//     }
//   }
// }

// Use the same function as in fluxes_calc_1 to verify the flux integral is
// correct
TEST(part_1, flux_integral_2) {
  constexpr int ctrl_vols_x = 2560, ctrl_vols_y = 2560;
  constexpr real min_x = 0.0, min_y = 0.0;
  constexpr real max_x = 1.0, max_y = 1.0;

  using MeshT = Mesh<ctrl_vols_x, ctrl_vols_y>;
  std::unique_ptr<MeshT> mesh =
      std::make_unique<MeshT>(min_x, max_x, min_y, max_y);
  EnergyAssembly<SecondOrderCentered_Part1> space_disc_diffuse(1.0, 0.0, 0.0,
                                                               1.0);
  EnergyAssembly<SecondOrderCentered_Part1> space_disc_u(1.0, 1.0, 0.0, 0.0);
  EnergyAssembly<SecondOrderCentered_Part1> space_disc_v(1.0, 0.0, 1.0, 0.0);
  EnergyAssembly<SecondOrderCentered_Part1> space_disc(1.0, 1.0, 1.0);

  TestUtils::fill_mesh(*mesh, [=](const real x, const real y) {
    return std::tuple<real, real, real>(space_disc.solution(x, y), 1.0, 1.0);
  });

  for(int i = 0; i < ctrl_vols_x - 2; i++) {
    for(int j = 0; j < ctrl_vols_y - 2; j++) {
      const real x     = mesh->x_median(i);
      const real y     = mesh->y_median(j);
      const real x_min = mesh->x_min(i);
      const real y_min = mesh->y_min(j);
      const real x_max = mesh->x_max(i);
      const real y_max = mesh->y_max(j);

      const real d2t_dx2 = (space_disc.solution_dx(x_max, y) -
                            space_disc.solution_dx(x_min, y)) /
                           mesh->dx();

      EXPECT_NEAR(d2t_dx2, space_disc.solution_dx2(x, y), 5e-6);

      const real d2t_dy2 = (space_disc.solution_dy(x, y_max) -
                            space_disc.solution_dy(x, y_min)) /
                           mesh->dy();

      EXPECT_NEAR(d2t_dy2, space_disc.solution_dy2(x, y), 5e-6);

      const real diffuse = (d2t_dx2 + d2t_dy2) / (reynolds * prandtl);
      const real dx_1    = space_disc_diffuse.dx_flux(*mesh, i, j);
      const real dx_2    = space_disc_diffuse.dx_flux(*mesh, i - 1, j);

      const real dy_1 = space_disc_diffuse.dy_flux(*mesh, i, j);
      const real dy_2 = space_disc_diffuse.dy_flux(*mesh, i, j - 1);
      printf(
          "%2d %2d (% .3e, % .3e): x2_deriv: (% .6e - % .6e) / % .3e = % .6e "
          "vs % .6e vs % .6e vs % .6e\n",
          i, j, x, y, dx_1, dx_2, mesh->dx(),
          (dx_1 - dx_2) / mesh->dx() + (dy_1 - dy_2) / mesh->dy(),
          space_disc_diffuse.solution_dx2(x, y) +
              space_disc_diffuse.solution_dy2(x, y),
          diffuse * reynolds * prandtl,
          space_disc_diffuse.flux_integral(*mesh, i, j));

      // EXPECT_NEAR(diffuse, space_disc_diffuse.flux_integral(*mesh, i, j),
      // 5e-5);

      // const real dt_dx =
      //     (space_disc_u.solution(x_max, y) * space_disc_u.u(x_max, y) -
      //      space_disc_u.solution(x_min, y) * space_disc_u.u(x_min, y)) /
      //     mesh->dx();
      // EXPECT_NEAR(dt_dx, space_disc.solution_dx(x, y) * space_disc.u(x, y),
      // 1e-5);

      // EXPECT_NEAR(dt_dx, -space_disc_u.flux_integral(*mesh, i, j), 1e-6);

      // const real dt_dy = space_disc_v.solution_dy(x, y) * space_disc_v.v(x,
      // y); EXPECT_NEAR(dt_dy, -space_disc_v.flux_integral(*mesh, i, j), 1e-6);

      // EXPECT_NEAR(space_disc.flux_integral(*mesh, i, j),
      //             -dt_dx - dt_dy + (d2t_dx2 + d2t_dy2) / (reynolds *
      //             prandtl), 5e-5);
    }
  }
}

constexpr int max_mesh_size = 2000;

template <typename MeshT>
typename std::enable_if<MeshT::x_dim() >= max_mesh_size, void>::type
compute_mesh_errors(std::vector<std::tuple<int, real, real, real> > &vec) {}

// Computes the error for a set of meshes which double in size.
// This stops once the mesh has an x dimension of 2000
template <typename MeshT>
typename std::enable_if<MeshT::x_dim() < max_mesh_size, void>::type
compute_mesh_errors(std::vector<std::tuple<int, real, real, real> > &vec) {
  EnergyAssembly<SecondOrderCentered_Part1> space_disc(1.0, 1.0, 1.0);
  constexpr int x_dim = MeshT::x_dim();
  constexpr int y_dim = MeshT::y_dim();

  constexpr real x_min = 0.0, y_min = 0.0;
  constexpr real x_max = 1.0, y_max = 1.0;
  {
    // Put mesh into it's own scope to ensure it's deallocated before the next
    // one is allocated
    auto mesh = std::make_unique<MeshT>(x_min, x_max, y_min, y_max);

    TestUtils::fill_mesh(*mesh, [=](const real x, const real y) {
      return std::tuple<real, real, real>(
          space_disc.solution(x, y), space_disc.u(x, y), space_disc.v(x, y));
    });

    const auto err = [=](const MeshT &mesh, const int i, const int j) {
      const real x = mesh.x_median(i);
      const real y = mesh.y_median(j);
      return space_disc.flux_integral(mesh, i, j) -
             -space_disc.flux_int_solution(x, y);
    };

    const real l1_err =
        TestUtils::l1_error<MeshT>(*mesh, err) / (x_dim * y_dim);
    const real l2_err   = TestUtils::l2_error<MeshT>(*mesh, err);
    const real linf_err = TestUtils::linf_error<MeshT>(*mesh, err);
    vec.push_back({x_dim, l1_err, l2_err, linf_err});
  }
  compute_mesh_errors<Mesh<x_dim * 2, y_dim * 2> >(vec);
}

TEST(part_1, flux_integral_convergence) {
  using err_tuple = std::tuple<int, real, real, real>;
  std::vector<err_tuple> errors;
  compute_mesh_errors<Mesh<10, 10> >(errors);
  real prev_errs[2][3] = {{q_nan, q_nan, q_nan}, {q_nan, q_nan, q_nan}};
  // For the solution to be correct, the order of convergence (as the mesh
  // grows) must match the order of the spatial discretization, and the
  // extrapolated estimate should go to 0
  for(const auto [m_dim, cur_l1, cur_l2, cur_linf] : errors) {
    if(!std::isnan(prev_errs[1][0])) {
      auto [order, extrap] =
          TestUtils::richardson(cur_l1, prev_errs[0][0], prev_errs[1][0]);
      EXPECT_NEAR(2.0, order, 2e-1);
      printf(
          "Richardson Estimated L1 Order: % .3e; Extrapolated: % .3e;"
          " for mesh %d\n",
          order, extrap, m_dim);
    }
    if(!std::isnan(prev_errs[1][1])) {
      auto [order, extrap] =
          TestUtils::richardson(cur_l2, prev_errs[0][1], prev_errs[1][1]);
      EXPECT_NEAR(2.0, order, 2e-1);
      printf(
          "Richardson Estimated L2 Order: % .3e; Extrapolated: % .3e;"
          " for mesh %d\n",
          order, extrap, m_dim);
    }
    if(!std::isnan(prev_errs[1][2])) {
      auto [order, extrap] =
          TestUtils::richardson(cur_linf, prev_errs[0][2], prev_errs[1][2]);
      EXPECT_NEAR(2.0, order, 1e-1);
      printf(
          "Richardson Estimated Linf Order: % .3e; Extrapolated: % .3e;"
          " for mesh %d\n",
          order, extrap, m_dim);
    }
    prev_errs[1][0] = prev_errs[0][0];
    prev_errs[1][1] = prev_errs[0][1];
    prev_errs[1][2] = prev_errs[0][2];

    prev_errs[0][0] = cur_l1;
    prev_errs[0][1] = cur_l2;
    prev_errs[0][2] = cur_linf;
    printf("Errors for %d x %d mesh: % .8e,   % .8e,  % .8e\n", m_dim, m_dim,
           cur_l1, cur_l2, cur_linf);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
