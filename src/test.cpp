
#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "test_utils.hpp"

using rngAlg = std::mt19937_64;

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}




