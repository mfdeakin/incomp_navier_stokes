
#include "boundaries.hpp"
#include "constants.hpp"
#include "mesh.hpp"
#include "time_disc.hpp"

#include <chrono>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

template <typename MeshT>
void plot_mesh_surface(const MeshT &mesh, std::string title_str) {
  // Imports needd for plotting the surface
  py::object figure = py::module::import("matplotlib.pyplot").attr("figure");
  py::object Axes3D = py::module::import("mpl_toolkits.mplot3d").attr("Axes3D");
  py::object cm     = py::module::import("matplotlib").attr("cm");
  py::object title  = py::module::import("matplotlib.pyplot").attr("title");

  std::vector<real> x_vals;
  std::vector<real> y_vals;
  std::vector<real> p_vals;
  std::vector<real> u_vals;
  std::vector<real> v_vals;

  constexpr int min_i = 0, max_i = 0;
  constexpr int min_j = 0, max_j = 0;
  // Aggregate the values for plotting
  for(int i = min_i; i < mesh.x_dim() - max_i; i++) {
    for(int j = min_j; j < mesh.y_dim() - max_j; j++) {
      x_vals.push_back(mesh.x_median(i));
      y_vals.push_back(mesh.y_median(j));
      p_vals.push_back(mesh.press(i, j));
      u_vals.push_back(mesh.u_vel(i, j));
      v_vals.push_back(mesh.v_vel(i, j));
    }
  }
  py::array x = py::array(x_vals.size(), x_vals.data())
                    .attr("reshape")(mesh.x_dim() - min_i - max_i,
                                     mesh.y_dim() - min_j - max_j);
  py::array y = py::array(y_vals.size(), y_vals.data())
                    .attr("reshape")(mesh.x_dim() - min_i - max_i,
                                     mesh.y_dim() - min_j - max_j);

  {
    py::array z = py::array(p_vals.size(), p_vals.data())
                      .attr("reshape")(mesh.x_dim() - min_i - max_i,
                                       mesh.y_dim() - min_j - max_j);
    py::object fig = figure();
    py::object ax  = fig.attr("gca")("projection"_a = "3d");
    ax.attr("plot_surface")(x, y, z, "cmap"_a = cm.attr("gist_heat"));
    title(("Pressure at " + title_str).c_str());
  }
  {
    py::array z = py::array(u_vals.size(), u_vals.data())
                      .attr("reshape")(mesh.x_dim() - min_i - max_i,
                                       mesh.y_dim() - min_j - max_j);

    py::object fig = figure();
    py::object ax  = fig.attr("gca")("projection"_a = "3d");
    ax.attr("plot_surface")(x, y, z, "cmap"_a = cm.attr("gist_heat"));
    title(("U Velocity at " + title_str).c_str());
  }
}

int main(int argc, char **argv) {
  // Our Python instance
  py::scoped_interpreter _{};
  py::object show = py::module::import("matplotlib.pyplot").attr("show");

  using RK = RK4_Solver<Mesh<160, 160>, SecondOrderCentered<BConds_Part1> >;
  BConds_Part1 bc(3.0, 0.1, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  RK rk_solver(bc);
  using IE =
      ImplicitEuler_Solver<Mesh<160, 160>, SecondOrderCentered<BConds_Part1> >;
  IE ie_solver(bc);

  std::stringstream ss;
  real prev_time = 0.0;
  ss.str("");
  ss << "Initial Mesh";
  plot_mesh_surface(rk_solver.mesh(), ss.str());
  while(rk_solver.time() < 0.1) {
    rk_solver.timestep(0.8);
    const real cur_time = rk_solver.time();
    if(cur_time - prev_time > 0.01) {
      ss.str("");
      ss << "RK Time: " << rk_solver.time();
      plot_mesh_surface(rk_solver.mesh(), ss.str());
      prev_time = cur_time;

			printf("Starting implicit euler timestep\n");
      ie_solver.timestep(rk_solver.time() - ie_solver.time());
			printf("Finished implicit euler timestep\n");
      ss.str("");
      ss << "IE Time: " << ie_solver.time();
      plot_mesh_surface(ie_solver.mesh(), ss.str());

      printf("%s\n", ss.str().c_str());
    }
  }

  ss.str("");
  ss << "RK4 Time: " << rk_solver.time();
  plot_mesh_surface(rk_solver.mesh(), ss.str());

  ss.str("");
  ss << "IE Time: " << ie_solver.time();
  plot_mesh_surface(ie_solver.mesh(), ss.str());

  show();
  return 0;
}
