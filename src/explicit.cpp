
#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "time_disc.hpp"

#include <string>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

template <typename Solver>
void plot_mesh_surface(const Solver &solver) {
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
      z_vals.push_back(mesh.Temp(i, j));
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

template <typename Solver>
void plot_mesh_contour(const Solver &solver) {
  py::object Figure  = py::module::import("matplotlib.pyplot").attr("figure");
  py::object Contour = py::module::import("matplotlib.pyplot").attr("contour");
  py::object Contourf =
      py::module::import("matplotlib.pyplot").attr("contourf");
  py::object CLabel = py::module::import("matplotlib.pyplot").attr("clabel");
  py::object cm     = py::module::import("matplotlib").attr("cm");
  py::object colorbar =
      py::module::import("matplotlib.pyplot").attr("colorbar");
  py::object arange = py::module::import("numpy").attr("arange");

  auto fig = Figure();

  const typename Solver::MeshT &mesh = solver.mesh();

  std::vector<real> x_vals;
  std::vector<real> y_vals;
  std::vector<real> z_vals;

  real min_z = std::numeric_limits<real>::infinity();
  real max_z = -std::numeric_limits<real>::infinity();

  for(int i = 0; i < mesh.x_dim(); i++) {
    for(int j = 0; j < mesh.y_dim(); j++) {
      x_vals.push_back(mesh.x_median(i));
      y_vals.push_back(mesh.y_median(j));
      z_vals.push_back(mesh.Temp(i, j));
      min_z = std::min(mesh.Temp(i, j), min_z);
      max_z = std::max(mesh.Temp(i, j), max_z);
    }
  }
  py::array x = py::array(x_vals.size(), x_vals.data())
                    .attr("reshape")(mesh.x_dim(), mesh.y_dim());
  py::array y = py::array(y_vals.size(), y_vals.data())
                    .attr("reshape")(mesh.x_dim(), mesh.y_dim());
  py::array z = py::array(z_vals.size(), z_vals.data())
                    .attr("reshape")(mesh.x_dim(), mesh.y_dim());

  Contourf(x, y, z, "cmap"_a = cm.attr("gist_heat"),
           "levels"_a = arange(min_z, max_z, (max_z - min_z) / 40.0));
  colorbar();
  auto cs = Contour(x, y, z);
  CLabel(cs);
}

// Look at norms of error in the flux
void plot_explicit_energy_evolution() {
  py::object Show = py::module::import("matplotlib.pyplot").attr("show");

  constexpr int ctrl_vols_x = 64;
  constexpr int ctrl_vols_y = 64;

  using MeshT     = Mesh<ctrl_vols_x, ctrl_vols_y>;
  using SpaceDisc = EnergyAssembly<SecondOrderCentered_Part1>;

  RK1_Solver<MeshT, SpaceDisc> solver;

  for(int i = 0; i < 14; i++) {
    plot_mesh_surface(solver);
    solver.timestep(0.25);
  }

  plot_mesh_surface(solver);
  Show();
}

void plot_implicit_energy_evolution() {
  py::object Show = py::module::import("matplotlib.pyplot").attr("show");

  constexpr int ctrl_vols_x = 256;
  constexpr int ctrl_vols_y = 256;

  using MeshT     = Mesh<ctrl_vols_x, ctrl_vols_y>;
  using SpaceDisc = EnergyAssembly<SecondOrderCentered_Part1>;

  ImplicitEuler_Solver<MeshT, SpaceDisc> solver;

  for(int i = 0; i < 14; i++) {
    plot_mesh_surface(solver);
    solver.timestep(0.1);
  }

  plot_mesh_surface(solver);
  Show();
}

void plot_dx_flux() {
  py::object Show = py::module::import("matplotlib.pyplot").attr("show");

  constexpr int ctrl_vols_x = 256;
  constexpr int ctrl_vols_y = 256;

  using MeshT         = Mesh<ctrl_vols_x, ctrl_vols_y>;
  using SpaceAssembly = EnergyAssembly<SecondOrderCentered_Part1>;
  using RK4           = RK4_Solver<MeshT, SpaceAssembly>;
  using RK4Flux = RK4_Solver<Mesh<ctrl_vols_x + 1, ctrl_vols_y>, SpaceAssembly>;
  RK4 solver;
  RK4Flux solver_fake;
  RK4Flux solver_fake_error;
  RK4Flux solver_fake_actual;
  SpaceAssembly &space_assembly = solver.space_assembly();
  MeshT &initial                = solver.mesh();

  for(int i = -1; i < initial.x_dim(); i++) {
    for(int j = 0; j < initial.y_dim(); j++) {
      const real x = initial.x_max(i);
      const real y = initial.y_median(j);

      solver_fake.mesh().Temp(i + 1, j) =
          space_assembly.dx_flux(initial, i, j, 0.0);
      solver_fake_actual.mesh().Temp(i + 1, j) =
          space_assembly.solution_dx(x, y, 0.0);
      solver_fake_error.mesh().Temp(i + 1, j) =
          space_assembly.dx_flux(initial, i, j, 0.0) -
          space_assembly.solution_dx(x, y, 0.0);
    }
  }

  py::object Title = py::module::import("matplotlib.pyplot").attr("title");
  // plot_mesh_contour(solver);
  // Title("Initial Conditions");

  // plot_mesh_contour(solver_fake);
  // Title("dT/dx FV");

  // plot_mesh_contour(solver_fake_actual);
  // Title("dT/dx Actual");

  // plot_mesh_contour(solver_fake_error);
  // Title("dT/dx Flux Error");

  plot_mesh_surface(solver_fake_error);
  Title("dT/dx Flux Error");
  Show();
}

void plot_dy_flux() {
  py::object Show = py::module::import("matplotlib.pyplot").attr("show");

  constexpr int ctrl_vols_x = 256;
  constexpr int ctrl_vols_y = 256;

  using MeshT         = Mesh<ctrl_vols_x, ctrl_vols_y>;
  using SpaceAssembly = EnergyAssembly<SecondOrderCentered_Part1>;
  using RK4           = RK4_Solver<MeshT, SpaceAssembly>;
  using RK4Flux = RK4_Solver<Mesh<ctrl_vols_x, ctrl_vols_y + 1>, SpaceAssembly>;
  RK4 solver;
  RK4Flux solver_fake;
  RK4Flux solver_fake_error;
  RK4Flux solver_fake_actual;
  SpaceAssembly &space_assembly = solver.space_assembly();
  MeshT &initial                = solver.mesh();

  for(int i = 0; i < initial.x_dim(); i++) {
    for(int j = -1; j < initial.y_dim(); j++) {
      const real x = initial.x_median(i);
      const real y = initial.y_max(j);

      solver_fake.mesh().Temp(i,
                              j + 1) =  // j + 1 because of the array boundaries
          space_assembly.dy_flux(
              initial, i, j,
              0.0);  // j because we're computing the flux at the boundaries
      solver_fake_actual.mesh().Temp(i, j + 1) =
          space_assembly.solution_dy(x, y, 0.0);
      solver_fake_error.mesh().Temp(i, j + 1) =
          space_assembly.dy_flux(initial, i, j, 0.0) -
          space_assembly.solution_dy(x, y, 0.0);
    }
  }

  py::object Title = py::module::import("matplotlib.pyplot").attr("title");
  // plot_mesh_contour(solver);
  // Title("Initial Conditions");

  // plot_mesh_contour(solver_fake);
  // Title("dT/dy FV");

  // plot_mesh_contour(solver_fake_actual);
  // Title("dT/dy Actual");

  // plot_mesh_contour(solver_fake_error);
  // Title("dT/dy Flux Error");

  plot_mesh_surface(solver_fake_error);
  Title("dT/dy Flux Error");
  Show();
}

void plot_nabla2_T() {
  py::object Show = py::module::import("matplotlib.pyplot").attr("show");

  constexpr int ctrl_vols_x = 256;
  constexpr int ctrl_vols_y = 256;

  using MeshT         = Mesh<ctrl_vols_x, ctrl_vols_y>;
  using SpaceAssembly = EnergyAssembly<SecondOrderCentered_Part1>;
  using RK4           = RK4_Solver<MeshT, SpaceAssembly>;
  RK4 solver;
  RK4 solver_fake;
  RK4 solver_fake_error;
  RK4 solver_fake_actual;
  SpaceAssembly &space_assembly = solver.space_assembly();
  MeshT &initial                = solver.mesh();

  for(int i = 0; i < initial.x_dim(); i++) {
    for(int j = 0; j < initial.y_dim(); j++) {
      const real x = initial.x_median(i);
      const real y = initial.y_median(j);

      solver_fake.mesh().Temp(i, j) =
          space_assembly.nabla2_T_flux_integral(initial, i, j, 0.0);
      solver_fake_actual.mesh().Temp(i, j) =
          space_assembly.flux_int_nabla2_T_sol(x, y, 0.0);
      solver_fake_error.mesh().Temp(i, j) =
          space_assembly.nabla2_T_flux_integral(initial, i, j, 0.0) -
          space_assembly.flux_int_nabla2_T_sol(x, y, 0.0);
    }
  }
  py::object Title = py::module::import("matplotlib.pyplot").attr("title");

  plot_mesh_contour(solver_fake);
  Title("nabla2 T FV");

  plot_mesh_contour(solver_fake_actual);
  Title("nabla2 T Actual");

  plot_mesh_contour(solver_fake_error);
  Title("nabla2 T Error");

  plot_mesh_surface(solver_fake_error);
  Title("nabla2 T Error");
  Show();
}

void plot_source() {
  py::object Show = py::module::import("matplotlib.pyplot").attr("show");

  constexpr int ctrl_vols_x = 256;
  constexpr int ctrl_vols_y = 256;

  using MeshT         = Mesh<ctrl_vols_x, ctrl_vols_y>;
  using SpaceAssembly = EnergyAssembly<SecondOrderCentered_Part1>;
  using RK4           = RK4_Solver<MeshT, SpaceAssembly>;
  RK4 solver;
  RK4 solver_fake;
  RK4 solver_fake_error;
  RK4 solver_fake_actual;
  SpaceAssembly &space_assembly = solver.space_assembly();
  MeshT &initial                = solver.mesh();

  for(int i = 0; i < initial.x_dim(); i++) {
    for(int j = 0; j < initial.y_dim(); j++) {
      const real x = initial.x_median(i);
      const real y = initial.y_median(j);

      solver_fake.mesh().Temp(i, j) =
          space_assembly.source_fd(initial, i, j, 0.0);
      solver_fake_actual.mesh().Temp(i, j) =
          space_assembly.source_sol(x, y, 0.0);
      solver_fake_error.mesh().Temp(i, j) =
          space_assembly.source_fd(initial, i, j, 0.0) -
          space_assembly.source_sol(x, y, 0.0);
    }
  }
  py::object Title = py::module::import("matplotlib.pyplot").attr("title");

  plot_mesh_contour(solver_fake);
  Title("Source FD");

  plot_mesh_contour(solver_fake_actual);
  Title("Source Actual");

  plot_mesh_contour(solver_fake_error);
  Title("Source Error");

  plot_mesh_surface(solver_fake_error);
  Title("Source Error");
  Show();
}

template <int mesh_dim>
void plot_flux_integral() {
  constexpr int ctrl_vols_x = mesh_dim;
  constexpr int ctrl_vols_y = mesh_dim;

  using MeshT         = Mesh<ctrl_vols_x, ctrl_vols_y>;
  using SpaceAssembly = EnergyAssembly<SecondOrderCentered_Part1>;
  using RK4           = RK4_Solver<MeshT, SpaceAssembly>;
  RK4 solver;
  RK4 solver_fake;
  RK4 solver_fake_error;
  RK4 solver_fake_actual;
  SpaceAssembly &space_assembly = solver.space_assembly();
  MeshT &initial                = solver.mesh();

  for(int i = 0; i < initial.x_dim(); i++) {
    for(int j = 0; j < initial.y_dim(); j++) {
      const real x = initial.x_median(i);
      const real y = initial.y_median(j);

      solver_fake.mesh().Temp(i, j) =
          space_assembly.flux_integral(initial, i, j, 0.0);
      solver_fake_actual.mesh().Temp(i, j) =
          space_assembly.flux_int_solution(x, y, 0.0);
      solver_fake_error.mesh().Temp(i, j) =
          space_assembly.flux_integral(initial, i, j, 0.0) -
          space_assembly.flux_int_solution(x, y, 0.0);
    }
  }
  py::object Title = py::module::import("matplotlib.pyplot").attr("title");

  std::stringstream ss;
  plot_mesh_contour(solver_fake);
  ss << "Flux Integral FV " << mesh_dim;
  Title(ss.str());

  ss.str(std::string());
  ss << "Flux Integral Actual " << mesh_dim;
  plot_mesh_contour(solver_fake_actual);
  Title(ss.str());

  ss.str(std::string());
  ss << "Flux Integral Error " << mesh_dim;
  plot_mesh_contour(solver_fake_error);
  Title(ss.str());

  plot_mesh_surface(solver_fake_error);
  Title(ss.str());
}

int main(int argc, char **argv) {
  // Our Python instance
  py::scoped_interpreter _{};

  py::object Show = py::module::import("matplotlib.pyplot").attr("show");

  // plot_source();
  plot_dx_flux();
  plot_dy_flux();
  plot_explicit_energy_evolution();
  plot_implicit_energy_evolution();
  // plot_nabla2_T();
  // plot_flux_integral<16>();
  // plot_flux_integral<32>();
  // plot_flux_integral<64>();
  // plot_flux_integral<128>();
  // Show();
  return 0;
}
