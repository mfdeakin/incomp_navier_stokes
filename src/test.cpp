
#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "test_utils.hpp"

using rngAlg = std::mt19937_64;

TEST(boundary_conds_part1, t0) {
  constexpr int ctrl_vols_x = 64, ctrl_vols_y = 64;
	constexpr real min_x = 0.0, min_y = 0.0;
	constexpr real max_x = 1.0, max_y = 1.0;
	constexpr real t_0 = 5.0 * pi;
  Mesh<ctrl_vols_x, ctrl_vols_y> mesh(min_x, max_x, min_y, max_y);
	SecondOrderCentered_Part1 space_disc(t_0);
	for(int i = 0; i < ctrl_vols_x; i++) {
		for(int j = 0; j < ctrl_vols_y; j++) {
			const real x = mesh.x_median(i);
			const real y = mesh.y_median(i);

			mesh(i, j) = space_disc.solution(x, y);
		}
	}
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
