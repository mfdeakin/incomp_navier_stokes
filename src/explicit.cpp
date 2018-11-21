
#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "time_disc.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

template <typename Solver>
void plot_mesh(const Solver &solver) {
  py::object Figure = py::module::import("matplotlib.pyplot").attr("figure");
  py::object Plot   = py::module::import("matplotlib.pyplot").attr("plot");
  py::object Title  = py::module::import("matplotlib.pyplot").attr("title");
  py::object Legend = py::module::import("matplotlib.pyplot").attr("legend");
  py::object Axes3D = py::module::import("mpl_toolkits.mplot3d").attr("Axes3D");
  py::object cm     = py::module::import("matplotlib").attr("cm");

  auto fig      = Figure();
  py::object ax = fig.attr("gca")("projection"_a = "3d");

  const typename Solver::MeshT &mesh = solver.mesh();

  std::vector<real> x_vals;
  std::vector<real> y_vals;
  std::vector<real> z_vals;

  for(int i = 0; i < mesh.x_dim(); i++) {
    for(int j = 0; j < mesh.y_dim(); j++) {
      x_vals.push_back(mesh.x_median(i));
      y_vals.push_back(mesh.y_median(j));
      z_vals.push_back(mesh.Temp(i, j) +
                       std::sin(2.0 * pi * mesh.x_median(i)) *
                           std::cos(2.0 * pi * mesh.y_median(j)));
    }
  }
  py::array x = py::array(x_vals.size(), x_vals.data())
                    .attr("reshape")(mesh.x_dim(), mesh.y_dim());
  py::array y = py::array(y_vals.size(), y_vals.data())
                    .attr("reshape")(mesh.x_dim(), mesh.y_dim());
  py::array z = py::array(z_vals.size(), z_vals.data())
                    .attr("reshape")(mesh.x_dim(), mesh.y_dim());

  ax.attr("plot_surface")(x, y, z, "cmap"_a = cm.attr("gist_heat"));
}

void plot_energy_evolution() {
  py::object Show = py::module::import("matplotlib.pyplot").attr("show");

  constexpr int ctrl_vols_x = 256;
  constexpr int ctrl_vols_y = 256;

  using MeshT     = Mesh<ctrl_vols_x, ctrl_vols_y>;
  using SpaceDisc = EnergyAssembly<SecondOrderCentered_Part1>;

  RK4_Solver<MeshT, SpaceDisc> solver;

  for(int i = 0; i < 6; i++) {
    plot_mesh(solver);
    Show();
    solver.timestep(0.25);
  }

  plot_mesh(solver);
  Show();
}

int main(int argc, char **argv) {
  // Our Python instance
  py::scoped_interpreter _{};
  plot_energy_evolution();
  return 0;
}
