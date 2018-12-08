
#include "boundaries.hpp"
#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "time_disc.hpp"

#include <sstream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// This just exports all of the objects we might need for

template <int ctrl_x, int ctrl_y>
py::class_<Mesh<ctrl_x, ctrl_y> > def_mesh(py::module &module) {
  using MeshT = Mesh<ctrl_x, ctrl_y>;
  std::stringstream ss;
  ss << "Mesh_" << ctrl_x << "x" << ctrl_y;
  py::class_<MeshT> mesh(module, ss.str().c_str());
  mesh.def(py::init<const BConds_Base *>())
      .def(py::init<real, real, real, real>(), py::arg("x_min") = 0.0,
           py::arg("x_max") = 1.0, py::arg("y_min") = 0.0,
           py::arg("y_max") = 1.0)
      .def("dx", &MeshT::dx)
      .def("dy", &MeshT::dy)
      .def("x_median", &MeshT::x_median)
      .def("y_median", &MeshT::y_median)
      .def("x_min", &MeshT::x_min)
      .def("y_min", &MeshT::y_min)
      .def("x_max", &MeshT::x_max)
      .def("y_max", &MeshT::y_max)
      .def("cell_idx", &MeshT::cell_idx)
      .def("pressure", (real & (MeshT::*)(int, int)) & MeshT::press)
      .def("u", (real & (MeshT::*)(int, int)) & MeshT::u_vel)
      .def("v", (real & (MeshT::*)(int, int)) & MeshT::v_vel);

  module.def("to_np_array", [](MeshT &m) {
    std::tuple<py::array, py::array, py::array> a{
        py::array(ctrl_x * ctrl_y, reinterpret_cast<real *>(m.pressure_data()))
            .attr("reshape")(ctrl_x, ctrl_y),
        py::array(ctrl_x * ctrl_y, reinterpret_cast<real *>(m.u_vel_data()))
            .attr("reshape")(ctrl_x, ctrl_y),
        py::array(ctrl_x * ctrl_y, reinterpret_cast<real *>(m.v_vel_data()))
            .attr("reshape")(ctrl_x, ctrl_y)};
    return a;
  });

  module.def("x_y_coords", [](const MeshT &m) {
    auto [x, y] = m.x_y_coords();
    std::pair<py::array, py::array> a{
        py::array(ctrl_x * ctrl_y, reinterpret_cast<real *>(&x))
            .attr("reshape")(ctrl_x, ctrl_y),
        py::array(ctrl_x * ctrl_y, reinterpret_cast<real *>(&y))
            .attr("reshape")(ctrl_x, ctrl_y)};
		return a;
  });

  return mesh;
}

template <typename BCond>
py::class_<BCond> def_bcond(py::module &module, const std::string &name) {
  py::class_<BCond> bc(module, name.c_str());
  bc.def(py::init<real, real, real, real, real, real, real, real>(),
         py::arg("P_0") = 1.0, py::arg("u_0") = 1.0, py::arg("v_0") = 1.0,
         py::arg("beta") = 1.0, py::arg("x_min") = 0.0, py::arg("x_max") = 1.0,
         py::arg("y_min") = 0.0, py::arg("y_max") = 1.0)
      .def("initial_conds", &BCond::initial_conds)
      .def("boundary_x_min", &BCond::boundary_x_min)
      .def("boundary_x_max", &BCond::boundary_x_max)
      .def("boundary_y_min", &BCond::boundary_y_min)
      .def("boundary_y_max", &BCond::boundary_y_max)
      .def("u", &BCond::u)
      .def("v", &BCond::v)
      // pybind11 seems to have a bug requiring the method pointer to be casted
      // to the inherited type
      .def("P_0", (real(BCond::*)()) & BCond::P_0)
      .def("u_0", (real(BCond::*)()) & BCond::u_0)
      .def("v_0", (real(BCond::*)()) & BCond::v_0)
      .def("beta", (real(BCond::*)()) & BCond::beta)
      .def("init_mesh", &BCond::template init_mesh<Mesh<10, 10> >)
      .def("init_mesh", &BCond::template init_mesh<Mesh<20, 20> >)
      .def("init_mesh", &BCond::template init_mesh<Mesh<40, 40> >)
      .def("init_mesh", &BCond::template init_mesh<Mesh<80, 80> >)
      .def("init_mesh", &BCond::template init_mesh<Mesh<160, 160> >)
      .def("init_mesh", &BCond::template init_mesh<Mesh<1024, 1024> >)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<10, 10> >)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<20, 20> >)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<40, 40> >)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<80, 80> >)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<160, 160> >)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<1024, 1024> >);

  return bc;
}

PYBIND11_MODULE(ins_solver, module) {
  def_bcond<BConds_Part1>(module, "BConds_Part1")
      .def("solution", &BConds_Part1::solution)
      .def("flux_int_solution", &BConds_Part1::flux_int_solution);

  def_mesh<10, 10>(module);
  def_mesh<20, 20>(module);
  def_mesh<40, 40>(module);
  def_mesh<80, 80>(module);
  def_mesh<160, 160>(module);
  def_mesh<1024, 1024>(module);

  using Assembly = INSAssembly<SecondOrderCentered>;
  py::class_<Assembly>(module, "INSAssembly")
      .def(py::init<const BConds_Part1 &>(), py::arg("boundaries"))
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<10, 10> >)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<20, 20> >)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<40, 40> >)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<80, 80> >)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<160, 160> >)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<1024, 1024> >);
}
