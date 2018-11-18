
#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "test_utils.hpp"

using rngAlg = std::mt19937_64;

// Verifies that the boundary flux calculations for x, y, dx, and dy are correct
// Uses the isotropic test function x * sin(pi * x) + y * sin(pi * y)
TEST(fluxes_calc_1, part_1) {
  constexpr real max_err    = 1e-6;
  constexpr int ctrl_vols_x = 128, ctrl_vols_y = 128;
  constexpr real min_x = 0.0, min_y = 0.0;
  constexpr real max_x = 0.1, max_y = 0.1;
  using MeshT = Mesh<ctrl_vols_x, ctrl_vols_y>;
  std::unique_ptr<MeshT> mesh =
      std::make_unique<MeshT>(min_x, max_x, min_y, max_y);
  EnergyAssembly<SecondOrderCentered_Part1> space_disc(q_nan, q_nan, q_nan);

  auto sol = [](const real x, const real y) {
    return (x * std::sin(pi * x)) + (y * std::sin(pi * y));
  };

  auto sol_dx = [](const real x, const real y) {
    return (x * pi * std::cos(pi * x) + std::sin(pi * x));
  };

  auto sol_dy = [=](const real x, const real y) { return sol_dx(y, x); };

  TestUtils::fill_mesh(*mesh, sol);

  // The flux computation assumes the boundary value,
  // so skip the last value which can't be computed from the mesh
  for(int i = 0; i < ctrl_vols_x - 1; i++) {
    for(int j = 0; j < ctrl_vols_y - 1; j++) {
      const real x = mesh->x_median(i);
      const real y = mesh->y_median(j);

      const real x_flux = sol(mesh->x_max(i), y);
      const real y_flux = sol(x, mesh->y_max(j));

      EXPECT_NEAR(x_flux, space_disc.x_flux(*mesh, i, j), max_err);
      EXPECT_NEAR(y_flux, space_disc.y_flux(*mesh, i, j), max_err);

      const real dx_flux = sol_dx(mesh->x_max(i), y);
      const real dy_flux = sol_dy(x, mesh->y_max(j));

      EXPECT_NEAR(dx_flux, space_disc.dx_flux(*mesh, i, j), max_err);
      EXPECT_NEAR(dy_flux, space_disc.dy_flux(*mesh, i, j), max_err);
    }
  }
}

// Verifies that the boundary flux calculations for x, y, dx, and dy are correct
// Uses the solution for part 1
TEST(fluxes_calc_2, part_1) {
  constexpr real max_err    = 5e-6;
  constexpr int ctrl_vols_x = 2560, ctrl_vols_y = 2560;
  constexpr real min_x = 0.0, min_y = 0.0;
  constexpr real max_x = 1.0, max_y = 1.0;
  using MeshT = Mesh<ctrl_vols_x, ctrl_vols_y>;
  std::unique_ptr<MeshT> mesh =
      std::make_unique<MeshT>(min_x, max_x, min_y, max_y);
  EnergyAssembly<SecondOrderCentered_Part1> space_disc(1.0, q_nan, q_nan);

  TestUtils::fill_mesh(*mesh, [=](const real x, const real y) {
    return space_disc.solution(x, y);
  });

  // Start at negative 1 to ensure the boundaries are implemented correctly
  for(int i = -1; i < ctrl_vols_x; i++) {
    for(int j = -1; j < ctrl_vols_y; j++) {
      const real x = mesh->x_median(i);
      const real y = mesh->y_median(j);

      // There isn't a well defined value when the coordinate of the dimension
      // the flux isn't being computed for is outside of the mesh
      if(j != -1) {
        const real x_max = mesh->x_max(i);
        EXPECT_NEAR(space_disc.solution(x_max, y),
                    space_disc.x_flux(*mesh, i, j), max_err);
        EXPECT_NEAR(space_disc.solution_dx(x_max, y),
                    space_disc.dx_flux(*mesh, i, j), max_err);
      }
      if(i != -1) {
        const real y_max = mesh->y_max(j);
        EXPECT_NEAR(space_disc.solution(x, y_max),
                    space_disc.y_flux(*mesh, i, j), max_err);
        EXPECT_NEAR(space_disc.solution_dy(x, y_max),
                    space_disc.dy_flux(*mesh, i, j), max_err);
      }
    }
  }
}

TEST(flux_integral_2, part_1) {
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
    return space_disc.solution(x, y);
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

      EXPECT_NEAR(diffuse, space_disc_diffuse.flux_integral(*mesh, i, j), 5e-5);

      const real dt_dx = space_disc_u.solution_dx(x, y) * space_disc_u.u(x, y);
      EXPECT_NEAR(dt_dx, -space_disc_u.flux_integral(*mesh, i, j), 1e-6);

      const real dt_dy = space_disc_v.solution_dy(x, y) * space_disc_v.v(x, y);
      EXPECT_NEAR(dt_dy, -space_disc_v.flux_integral(*mesh, i, j), 1e-6);

      EXPECT_NEAR(space_disc.flux_integral(*mesh, i, j),
                  -dt_dx - dt_dy + (d2t_dx2 + d2t_dy2) / (reynolds * prandtl),
                  5e-5);
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
