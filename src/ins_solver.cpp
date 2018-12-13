
#include "boundaries.hpp"
#include "constants.hpp"
#include "mesh.hpp"
#include "space_disc.hpp"
#include "time_disc.hpp"

#include <sstream>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// This just exports all of the objects we might need for

template <int ctrl_x, int ctrl_y>
py::class_<Mesh<ctrl_x, ctrl_y>> def_mesh(py::module &module) {
  using MeshT = Mesh<ctrl_x, ctrl_y>;
  std::stringstream ss;
  ss << "Mesh_" << ctrl_x << "x" << ctrl_y;
  py::class_<MeshT> mesh(module, ss.str().c_str());
  mesh.def(py::init<const BConds_Part1 &>())
      .def(py::init<real, real, real, real>(), py::arg("x_min") = 0.0,
           py::arg("x_max") = 1.0, py::arg("y_min") = 0.0,
           py::arg("y_max") = 1.0)
      .def("y_dim", &MeshT::x_dim)
      .def("x_dim", &MeshT::y_dim)
      .def("dx", &MeshT::dx)
      .def("dy", &MeshT::dy)
      .def("x_median", &MeshT::x_median)
      .def("y_median", &MeshT::y_median)
      .def("x_min", &MeshT::x_min)
      .def("y_min", &MeshT::y_min)
      .def("x_max", &MeshT::x_max)
      .def("y_max", &MeshT::y_max)
      .def("cell_idx", &MeshT::cell_idx)
      .def("__getitem__", &MeshT::operator())
      .def("__getitem__",
           [](MeshT &mesh, std::pair<int, int> c) {
             return mesh(c.first, c.second);
           })
      .def("__setitem__",
           [](MeshT &mesh, int i, int j, const triple &t) {
             mesh.press(i, j) = t(0);
             mesh.u_vel(i, j) = t(1);
             mesh.v_vel(i, j) = t(2);
             return mesh;
           })
      .def("__setitem__",
           [](MeshT &mesh, const std::pair<int, int> &c, const triple &t) {
             mesh.press(c.first, c.second) = t(0);
             mesh.u_vel(c.first, c.second) = t(1);
             mesh.v_vel(c.first, c.second) = t(2);
             return mesh;
           })
      .def("pressure", (real & (MeshT::*)(int, int)) & MeshT::press)
      .def("u", (real & (MeshT::*)(int, int)) & MeshT::u_vel)
      .def("v", (real & (MeshT::*)(int, int)) & MeshT::v_vel)
      .def("set_pressure",
           [](MeshT &m, int i, int j, real val) { m.press(i, j) = val; })
      .def("set_u",
           [](MeshT &m, int i, int j, real val) { m.u_vel(i, j) = val; })
      .def("set_v",
           [](MeshT &m, int i, int j, real val) { m.v_vel(i, j) = val; });

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

  using SpaceAssembly = INSAssembly<SecondOrderCentered<BConds_Part1>>;
  using RK1           = RK1_Solver<MeshT, SpaceAssembly>;
  ss.str("");
  ss << "RK1_" << ctrl_x << "x" << ctrl_y;
  py::class_<RK1> rk1(module, ss.str().c_str());
  rk1.def(py::init<const BConds_Part1 &>(), py::arg("boundaries"),
          py::return_value_policy::reference)
      .def("timestep", (void (RK1::*)(real)) & RK1::timestep)
      .def("time", (real(RK1::*)()) & RK1::time)
      .def("mesh", (MeshT & (RK1::*)()) & RK1::mesh)
      .def("space_assembly", (SpaceAssembly & (RK1::*)()) & RK1::space_assembly,
           py::return_value_policy::reference_internal);
  // The reference_internal rvp indicates the reference's lifetime is tied to
  // the timestep object. Basically it indicates I shouldn't have mixed static
  // and dynamic polymorphism at this level...
  return mesh;
}

template <typename BCond>
py::class_<BCond> def_bcond(py::module &module, const std::string &name) {
  py::class_<BCond> bc(module, name.c_str());
  bc.def(py::init<real, real, real, real, real, real, real, real, real>(),
         py::arg("P_0") = 1.0, py::arg("u_0") = 1.0, py::arg("v_0") = 1.0,
         py::arg("beta") = 1.0, py::arg("reynolds") = 1.0,
         py::arg("x_min") = 0.0, py::arg("x_max") = 1.0, py::arg("y_min") = 0.0,
         py::arg("y_max") = 1.0)
      .def("x_min", (real(BCond::*)()) & BCond::x_min)
      .def("x_max", (real(BCond::*)()) & BCond::x_max)
      .def("y_min", (real(BCond::*)()) & BCond::y_min)
      .def("y_max", (real(BCond::*)()) & BCond::y_max)
      .def("initial_conds",
           (std::tuple<real, real, real>(BCond::*)(real, real, real, real)) &
               BCond::initial_conds)
      .def("P_0", (real(BCond::*)()) & BCond::P_0)
      .def("u_0", (real(BCond::*)()) & BCond::u_0)
      .def("v_0", (real(BCond::*)()) & BCond::v_0)
      .def("beta", (real(BCond::*)()) & BCond::beta)
      .def("init_mesh", (void (BCond::*)(Mesh<10, 10> &)) &
                            BCond::template init_mesh<Mesh<10, 10>>)
      .def("init_mesh", (void (BCond::*)(Mesh<20, 20> &)) &
                            BCond::template init_mesh<Mesh<20, 20>>)
      .def("init_mesh", (void (BCond::*)(Mesh<40, 40> &)) &
                            BCond::template init_mesh<Mesh<40, 40>>)
      .def("init_mesh", (void (BCond::*)(Mesh<80, 80> &)) &
                            BCond::template init_mesh<Mesh<80, 80>>)
      .def("init_mesh", (void (BCond::*)(Mesh<160, 160> &)) &
                            BCond::template init_mesh<Mesh<160, 160>>)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<10, 10>>)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<20, 20>>)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<40, 40>>)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<80, 80>>)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<160, 160>>)
      .def("flux_soln", &BCond::template flux_int_fill<Mesh<1024, 1024>>);

  return bc;
}

std::pair<py::class_<triple>, py::class_<Jacobian>> def_multivar(
    py::module &module) {
  py::class_<triple> t(module, "triple");
  t.def(py::init<>())
      .def(py::init<>([](const real t1, const real t2, const real t3) {
        return triple{t1, t2, t3};
      }));
  using t_getter_type       = real &(triple::*)(int);
  const t_getter_type f_ptr = &triple::template operator()<int>;
  t.def("get", f_ptr, py::arg("i"))
      .def("__getitem__", f_ptr, py::arg("i"))
      .def("__setitem__",
           [](triple &t, int i, real v) {
             t(i) = v;
             return t;
           },
           py::arg("i"), py::arg("v"))
      .def(-py::self)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self += py::self)
      .def(py::self * py::self)
      .def(py::self * real())
      .def(real() * py::self)
      .def(py::self *= real())
      .def(py::self / real())
      .def(py::self /= real())
      .def("__len__", [](const triple &t) { return t.extent(0); })
      .def("__str__", [](const triple &t) {
        std::stringstream ss;
        for(int i = 0; i < t.extent(0); i++) {
          ss << t(i) << std::endl;
        }
        return ss.str();
      });

  py::class_<Jacobian> j(module, "Jacobian");
  j.def(py::init<>());
  using j_r_getter_type             = real &(Jacobian::*)(int, int);
  const j_r_getter_type j_r_ptr     = &Jacobian::operator();
  using j_t_getter_type             = triple &(Jacobian::*)(int);
  const j_t_getter_type row_ptr     = &Jacobian::row;
  using j_t_val_getter_type         = triple (Jacobian::*)(int) const;
  const j_t_val_getter_type col_ptr = &Jacobian::column;
  j.def("__getitem__", j_r_ptr, py::arg("i"), py::arg("j"))
      .def("__getitem__",
           [](Jacobian &J, std::pair<int, int> c) {
             return J(c.first, c.second);
           })
      .def("__setitem__",
           [](Jacobian &J, int i, int j, real v) {
             J(i, j) = v;
             return J;
           },
           py::arg("i"), py::arg("j"), py::arg("v"))
      .def("__setitem__",
           [](Jacobian &J, const std::pair<int, int> &c, const real r) {
             J(c.first, c.second) = r;
             return J;
           })
      .def("__str__",
           [](const Jacobian &J) {
             std::stringstream ss;
             for(int i = 0; i < J.extent(0); i++) {
               for(int j = 0; j < J.extent(1); j++) {
                 ss << J(i, j) << ", ";
               }
               ss << std::endl;
             }
             return ss.str();
           })
      .def("inverse", &Jacobian::inverse)
      .def("minor", &Jacobian::minor)
      .def("minors", &Jacobian::minors)
      .def("det", &Jacobian::det)
      .def("row", row_ptr, py::arg("r"))
      .def("column", col_ptr, py::arg("c"))
      .def(-py::self)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self * triple())
      .def(py::self *= py::self)
      .def(py::self * py::self)
      .def(py::self *= real())
      .def(py::self * real());
  return {t, j};
}

PYBIND11_MODULE(ins_solver, module) {
  def_multivar(module);

  def_bcond<BConds_Part1>(module, "BConds_Part1")
      .def("u", &BConds_Part1::u)
      .def("v", &BConds_Part1::v)
      .def("solution", &BConds_Part1::solution)
      .def("flux_int_solution", &BConds_Part1::flux_int_solution);

  def_mesh<10, 10>(module);
  def_mesh<20, 20>(module);
  def_mesh<40, 40>(module);
  def_mesh<80, 80>(module);
  def_mesh<160, 160>(module);
  def_mesh<1024, 1024>(module);

  using Assembly = INSAssembly<SecondOrderCentered<BConds_Part1>>;
  py::class_<Assembly>(module, "INSAssembly")
      .def(py::init<const BConds_Part1 &>(), py::arg("boundaries"))
      // .def("boundaries",
      //      [](Assembly &self) {
      //        printf("self: %p\n", &self);
      //        return self.boundaries();
      //      })
      .def("boundaries",
           (BConds_Part1 & (Assembly::*)()) & Assembly::boundaries)
      .def("flux_integral", &Assembly::flux_integral<Mesh<10, 10>>)
      .def("flux_integral", &Assembly::flux_integral<Mesh<20, 20>>)
      .def("flux_integral", &Assembly::flux_integral<Mesh<40, 40>>)
      .def("flux_integral", &Assembly::flux_integral<Mesh<80, 80>>)
      .def("flux_integral", &Assembly::flux_integral<Mesh<160, 160>>)
      .def("flux_integral", &Assembly::flux_integral<Mesh<1024, 1024>>)

      .def("flux_assembly", &Assembly::flux_assembly<Mesh<10, 10>>)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<20, 20>>)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<40, 40>>)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<80, 80>>)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<160, 160>>)
      .def("flux_assembly", &Assembly::flux_assembly<Mesh<1024, 1024>>)

      .def("jacobian_x_p1",
           (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<10, 10>>)
      .def("jacobian_x_p1",
           (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<20, 20>>)
      .def("jacobian_x_p1",
           (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<40, 40>>)
      .def("jacobian_x_p1",
           (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<80, 80>>)
      .def("jacobian_x_p1",
           (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<160, 160>>)
      .def("jacobian_x_p1",
           (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<1024, 1024>>)

      .def("jacobian_x_0",
           (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
               Assembly::jacobian_x_0<Mesh<10, 10>>)
      .def("jacobian_x_0",
           (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
               Assembly::jacobian_x_0<Mesh<20, 20>>)
      .def("jacobian_x_0",
           (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
               Assembly::jacobian_x_0<Mesh<40, 40>>)
      .def("jacobian_x_0",
           (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
               Assembly::jacobian_x_0<Mesh<80, 80>>)
      .def("jacobian_x_0",
           (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
               Assembly::jacobian_x_0<Mesh<160, 160>>)
      .def("jacobian_x_0",
           (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
               Assembly::jacobian_x_0<Mesh<1024, 1024>>)

      .def("jacobian_y_p1",
           (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<10, 10>>)
      .def("jacobian_y_p1",
           (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<20, 20>>)
      .def("jacobian_y_p1",
           (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<40, 40>>)
      .def("jacobian_y_p1",
           (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<80, 80>>)
      .def("jacobian_y_p1",
           (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<160, 160>>)
      .def("jacobian_y_p1",
           (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
               Assembly::jacobian_x_p1<Mesh<1024, 1024>>)

      .def("jacobian_y_0",
           (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
               Assembly::jacobian_y_0<Mesh<10, 10>>)
      .def("jacobian_y_0",
           (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
               Assembly::jacobian_y_0<Mesh<20, 20>>)
      .def("jacobian_y_0",
           (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
               Assembly::jacobian_y_0<Mesh<40, 40>>)
      .def("jacobian_y_0",
           (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
               Assembly::jacobian_y_0<Mesh<80, 80>>)
      .def("jacobian_y_0",
           (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
               Assembly::jacobian_y_0<Mesh<160, 160>>)
      .def("jacobian_y_0",
           (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
               Assembly::jacobian_y_0<Mesh<1024, 1024>>)

      .def("Dx_p1", (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
                        Assembly::Dx_p1<Mesh<10, 10>>)
      .def("Dx_p1", (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
                        Assembly::Dx_p1<Mesh<20, 20>>)
      .def("Dx_p1", (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
                        Assembly::Dx_p1<Mesh<40, 40>>)
      .def("Dx_p1", (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
                        Assembly::Dx_p1<Mesh<80, 80>>)
      .def("Dx_p1", (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
                        Assembly::Dx_p1<Mesh<160, 160>>)
      .def("Dx_p1",
           (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
               Assembly::Dx_p1<Mesh<1024, 1024>>)

      .def("Dx_0", (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
                       Assembly::Dx_0<Mesh<10, 10>>)
      .def("Dx_0", (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
                       Assembly::Dx_0<Mesh<20, 20>>)
      .def("Dx_0", (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
                       Assembly::Dx_0<Mesh<40, 40>>)
      .def("Dx_0", (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
                       Assembly::Dx_0<Mesh<80, 80>>)
      .def("Dx_0", (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
                       Assembly::Dx_0<Mesh<160, 160>>)
      .def("Dx_0", (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
                       Assembly::Dx_0<Mesh<1024, 1024>>)

      .def("Dx_m1", (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
                        Assembly::Dx_m1<Mesh<10, 10>>)
      .def("Dx_m1", (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
                        Assembly::Dx_m1<Mesh<20, 20>>)
      .def("Dx_m1", (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
                        Assembly::Dx_m1<Mesh<40, 40>>)
      .def("Dx_m1", (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
                        Assembly::Dx_m1<Mesh<80, 80>>)
      .def("Dx_m1", (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
                        Assembly::Dx_m1<Mesh<160, 160>>)
      .def("Dx_m1",
           (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
               Assembly::Dx_m1<Mesh<1024, 1024>>)

      .def("Dy_p1", (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
                        Assembly::Dy_p1<Mesh<10, 10>>)
      .def("Dy_p1", (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
                        Assembly::Dy_p1<Mesh<20, 20>>)
      .def("Dy_p1", (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
                        Assembly::Dy_p1<Mesh<40, 40>>)
      .def("Dy_p1", (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
                        Assembly::Dy_p1<Mesh<80, 80>>)
      .def("Dy_p1", (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
                        Assembly::Dy_p1<Mesh<160, 160>>)
      .def("Dy_p1",
           (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
               Assembly::Dy_p1<Mesh<1024, 1024>>)

      .def("Dy_0", (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
                       Assembly::Dy_0<Mesh<10, 10>>)
      .def("Dy_0", (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
                       Assembly::Dy_0<Mesh<20, 20>>)
      .def("Dy_0", (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
                       Assembly::Dy_0<Mesh<40, 40>>)
      .def("Dy_0", (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
                       Assembly::Dy_0<Mesh<80, 80>>)
      .def("Dy_0", (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
                       Assembly::Dy_0<Mesh<160, 160>>)
      .def("Dy_0", (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
                       Assembly::Dy_0<Mesh<1024, 1024>>)

      .def("Dy_m1", (Jacobian(Assembly::*)(Mesh<10, 10> &, int, int, real)) &
                        Assembly::Dy_m1<Mesh<10, 10>>)
      .def("Dy_m1", (Jacobian(Assembly::*)(Mesh<20, 20> &, int, int, real)) &
                        Assembly::Dy_m1<Mesh<20, 20>>)
      .def("Dy_m1", (Jacobian(Assembly::*)(Mesh<40, 40> &, int, int, real)) &
                        Assembly::Dy_m1<Mesh<40, 40>>)
      .def("Dy_m1", (Jacobian(Assembly::*)(Mesh<80, 80> &, int, int, real)) &
                        Assembly::Dy_m1<Mesh<80, 80>>)
      .def("Dy_m1", (Jacobian(Assembly::*)(Mesh<160, 160> &, int, int, real)) &
                        Assembly::Dy_m1<Mesh<160, 160>>)
      .def("Dy_m1",
           (Jacobian(Assembly::*)(Mesh<1024, 1024> &, int, int, real)) &
               Assembly::Dy_m1<Mesh<1024, 1024>>);
}
