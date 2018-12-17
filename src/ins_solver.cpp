
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

  using SpaceDisc     = SecondOrderCentered<BConds_Part1>;
  using Base_Solver_1 = Base_Solver<MeshT, SpaceDisc>;
  ss.str("");
  ss << "Base_1_" << ctrl_x << "x" << ctrl_y;
  py::class_<Base_Solver_1> base_1(module, ss.str().c_str());
  base_1.def(py::init<const BConds_Part1 &>(), py::arg("boundaries"))
      .def("time", &Base_Solver_1::time)
      .def("space_assembly", &Base_Solver_1::space_assembly)
      .def("mesh",
           (const MeshT &(Base_Solver_1::*)() const) & Base_Solver_1::mesh)
      .def("mesh", (MeshT & (Base_Solver_1::*)()) & Base_Solver_1::mesh);

  using RK1 = RK1_Solver<MeshT, SpaceDisc>;
  ss.str("");
  ss << "RK1_" << ctrl_x << "x" << ctrl_y;
  py::class_<RK1, Base_Solver_1> rk1(module, ss.str().c_str());
  rk1.def(py::init<const BConds_Part1 &>(), py::arg("boundaries"))
      .def("timestep", &RK1::timestep);

  using RK4 = RK4_Solver<MeshT, SpaceDisc>;
  ss.str("");
  ss << "RK4_" << ctrl_x << "x" << ctrl_y;
  py::class_<RK4, Base_Solver_1> rk4(module, ss.str().c_str());
  rk4.def(py::init<const BConds_Part1 &>(), py::arg("boundaries"))
      .def("timestep", &RK4::timestep);

  using SpaceDisc3    = SecondOrderCentered<BConds_Part3>;
  using Base_Solver_3 = Base_Solver<MeshT, SpaceDisc3>;
  ss.str("");
  ss << "Base_3_" << ctrl_x << "x" << ctrl_y;
  py::class_<Base_Solver_3> base_3(module, ss.str().c_str());
  base_3.def(py::init<const BConds_Part3 &>(), py::arg("boundaries"))
      .def("time", &Base_Solver_3::time)
      .def("space_assembly", &Base_Solver_3::space_assembly)
      .def("mesh",
           (const MeshT &(Base_Solver_3::*)() const) & Base_Solver_3::mesh)
      .def("mesh", (MeshT & (Base_Solver_3::*)()) & Base_Solver_3::mesh);

  using IE = ImplicitEuler_Solver<MeshT, SpaceDisc3>;
  ss.str("");
  ss << "IE_" << ctrl_x << "x" << ctrl_y;
  py::class_<IE, Base_Solver_3> ie(module, ss.str().c_str());
  ie.def(py::init<const BConds_Part3 &>(), py::arg("boundaries"))
      .def("timestep", &IE::timestep);
  return mesh;
}

template <typename BCond>
py::class_<BCond> def_bcond(py::module &module, const std::string &name) {
  py::class_<BConds_Base<BCond>> bc_base(module, ("base" + name).c_str());
  bc_base
      .def(py::init<real, real, real, real, real, real, real, real, real>(),
           py::arg("P_0") = 1.0, py::arg("u_0") = 1.0, py::arg("v_0") = 1.0,
           py::arg("beta") = 1.0, py::arg("reynolds") = 1.0,
           py::arg("x_min") = 0.0, py::arg("x_max") = 1.0,
           py::arg("y_min") = 0.0, py::arg("y_max") = 1.0)
      .def("x_min", &BCond::x_min)
      .def("x_max", &BCond::x_max)
      .def("y_min", &BCond::y_min)
      .def("y_max", &BCond::y_max)
      .def("initial_conds", &BCond::initial_conds)
      .def("P_0", &BCond::P_0)
      .def("u_0", &BCond::u_0)
      .def("v_0", &BCond::v_0)
      .def("beta", &BCond::beta)
      .def("init_mesh", &BCond::template init_mesh<Mesh<10, 10>>)
      .def("init_mesh", &BCond::template init_mesh<Mesh<20, 20>>)
      .def("init_mesh", &BCond::template init_mesh<Mesh<40, 40>>)
      .def("init_mesh", &BCond::template init_mesh<Mesh<80, 80>>)
      .def("init_mesh", &BCond::template init_mesh<Mesh<160, 160>>);

  py::class_<BCond, BConds_Base<BCond>> bc(module, name.c_str());

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
      .def(py::self * py::self)
      .def(py::self * real())
      .def(real() * py::self)
      .def(py::self *= real())
      .def(py::self / real())
      .def(py::self /= real())
      .def("__len__", [](const triple &t) { return t.extent(0); })
      .def("__str__", [](const triple &t) {
        std::stringstream ss;
        ss.precision(12);
        for(int i = 0; i < t.extent(0); i++) {
          ss << std::scientific << t(i) << std::endl;
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
             ss.precision(12);
             for(int i = 0; i < J.extent(0); i++) {
               for(int j = 0; j < J.extent(1); j++) {
                 ss << std::scientific << J(i, j) << ", ";
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
      .def(py::init<real, real, real, real, real, real, real, real, real>(),
           py::arg("P_0") = 1.0, py::arg("u_0") = 1.0, py::arg("v_0") = 1.0,
           py::arg("beta") = 1.0, py::arg("reynolds") = 1.0,
           py::arg("x_min") = 0.0, py::arg("x_max") = 1.0,
           py::arg("y_min") = 0.0, py::arg("y_max") = 1.0)
      .def("flux_soln", &BConds_Part1::template flux_int_fill<Mesh<10, 10>>)
      .def("flux_soln", &BConds_Part1::template flux_int_fill<Mesh<20, 20>>)
      .def("flux_soln", &BConds_Part1::template flux_int_fill<Mesh<40, 40>>)
      .def("flux_soln", &BConds_Part1::template flux_int_fill<Mesh<80, 80>>)
      .def("flux_soln", &BConds_Part1::template flux_int_fill<Mesh<160, 160>>)
      .def("flux_soln", &BConds_Part1::template flux_int_fill<Mesh<1024, 1024>>)
      .def("u", &BConds_Part1::u)
      .def("v", &BConds_Part1::v)
      .def("solution", &BConds_Part1::solution)
      .def("flux_int_solution", &BConds_Part1::flux_int_solution);

  def_bcond<BConds_Part3>(module, "BConds_Part3")
      .def(py::init<real, real, real, real, real, real, real, real, real,
                    real>(),
           py::arg("wall_vel") = 0.0, py::arg("P_0") = 1.0,
           py::arg("u_0") = 1.0, py::arg("v_0") = 1.0, py::arg("beta") = 1.0,
           py::arg("reynolds") = 1.0, py::arg("x_min") = 0.0,
           py::arg("x_max") = 1.0, py::arg("y_min") = 0.0,
           py::arg("y_max") = 1.0);

  def_mesh<10, 10>(module);
  def_mesh<20, 20>(module);
  def_mesh<40, 40>(module);
  def_mesh<80, 80>(module);
  def_mesh<160, 160>(module);
  def_mesh<1024, 1024>(module);

  using Discretization = SecondOrderCentered<BConds_Part1>;
  py::class_<Discretization>(module, "SecondOrderCentered")
      .def(py::init<const BConds_Part1 &>(), py::arg("boundaries"))
      .def("Dx_p1", &Discretization::Dx_p1<Mesh<10, 10>>)
      .def("Dx_p1", &Discretization::Dx_p1<Mesh<20, 20>>)
      .def("Dx_p1", &Discretization::Dx_p1<Mesh<40, 40>>)
      .def("Dx_p1", &Discretization::Dx_p1<Mesh<80, 80>>)
      .def("Dx_p1", &Discretization::Dx_p1<Mesh<160, 160>>)
      .def("Dx_p1", &Discretization::Dx_p1<Mesh<1024, 1024>>)

      .def("Dx_0", &Discretization::Dx_0<Mesh<10, 10>>)
      .def("Dx_0", &Discretization::Dx_0<Mesh<20, 20>>)
      .def("Dx_0", &Discretization::Dx_0<Mesh<40, 40>>)
      .def("Dx_0", &Discretization::Dx_0<Mesh<80, 80>>)
      .def("Dx_0", &Discretization::Dx_0<Mesh<160, 160>>)
      .def("Dx_0", &Discretization::Dx_0<Mesh<1024, 1024>>)

      .def("Dx_m1", &Discretization::Dx_m1<Mesh<10, 10>>)
      .def("Dx_m1", &Discretization::Dx_m1<Mesh<20, 20>>)
      .def("Dx_m1", &Discretization::Dx_m1<Mesh<40, 40>>)
      .def("Dx_m1", &Discretization::Dx_m1<Mesh<80, 80>>)
      .def("Dx_m1", &Discretization::Dx_m1<Mesh<160, 160>>)
      .def("Dx_m1", &Discretization::Dx_m1<Mesh<1024, 1024>>)

      .def("Dy_p1", &Discretization::Dy_p1<Mesh<10, 10>>)
      .def("Dy_p1", &Discretization::Dy_p1<Mesh<20, 20>>)
      .def("Dy_p1", &Discretization::Dy_p1<Mesh<40, 40>>)
      .def("Dy_p1", &Discretization::Dy_p1<Mesh<80, 80>>)
      .def("Dy_p1", &Discretization::Dy_p1<Mesh<160, 160>>)
      .def("Dy_p1", &Discretization::Dy_p1<Mesh<1024, 1024>>)

      .def("Dy_0", &Discretization::Dy_0<Mesh<10, 10>>)
      .def("Dy_0", &Discretization::Dy_0<Mesh<20, 20>>)
      .def("Dy_0", &Discretization::Dy_0<Mesh<40, 40>>)
      .def("Dy_0", &Discretization::Dy_0<Mesh<80, 80>>)
      .def("Dy_0", &Discretization::Dy_0<Mesh<160, 160>>)
      .def("Dy_0", &Discretization::Dy_0<Mesh<1024, 1024>>)

      .def("Dy_m1", &Discretization::Dy_m1<Mesh<10, 10>>)
      .def("Dy_m1", &Discretization::Dy_m1<Mesh<20, 20>>)
      .def("Dy_m1", &Discretization::Dy_m1<Mesh<40, 40>>)
      .def("Dy_m1", &Discretization::Dy_m1<Mesh<80, 80>>)
      .def("Dy_m1", &Discretization::Dy_m1<Mesh<160, 160>>)
      .def("Dy_m1", &Discretization::Dy_m1<Mesh<1024, 1024>>)

      .def("press_x_flux", &Discretization::press_x_flux<Mesh<10, 10>>)
      .def("press_x_flux", &Discretization::press_x_flux<Mesh<20, 20>>)
      .def("press_x_flux", &Discretization::press_x_flux<Mesh<40, 40>>)
      .def("press_x_flux", &Discretization::press_x_flux<Mesh<80, 80>>)
      .def("press_x_flux", &Discretization::press_x_flux<Mesh<160, 160>>)
      .def("press_x_flux", &Discretization::press_x_flux<Mesh<1024, 1024>>)

      .def("press_y_flux", &Discretization::press_y_flux<Mesh<10, 10>>)
      .def("press_y_flux", &Discretization::press_y_flux<Mesh<20, 20>>)
      .def("press_y_flux", &Discretization::press_y_flux<Mesh<40, 40>>)
      .def("press_y_flux", &Discretization::press_y_flux<Mesh<80, 80>>)
      .def("press_y_flux", &Discretization::press_y_flux<Mesh<160, 160>>)
      .def("press_y_flux", &Discretization::press_y_flux<Mesh<1024, 1024>>)

      .def("u_x_flux", &Discretization::u_x_flux<Mesh<10, 10>>)
      .def("u_x_flux", &Discretization::u_x_flux<Mesh<20, 20>>)
      .def("u_x_flux", &Discretization::u_x_flux<Mesh<40, 40>>)
      .def("u_x_flux", &Discretization::u_x_flux<Mesh<80, 80>>)
      .def("u_x_flux", &Discretization::u_x_flux<Mesh<160, 160>>)
      .def("u_x_flux", &Discretization::u_x_flux<Mesh<1024, 1024>>)

      .def("u_y_flux", &Discretization::u_y_flux<Mesh<10, 10>>)
      .def("u_y_flux", &Discretization::u_y_flux<Mesh<20, 20>>)
      .def("u_y_flux", &Discretization::u_y_flux<Mesh<40, 40>>)
      .def("u_y_flux", &Discretization::u_y_flux<Mesh<80, 80>>)
      .def("u_y_flux", &Discretization::u_y_flux<Mesh<160, 160>>)
      .def("u_y_flux", &Discretization::u_y_flux<Mesh<1024, 1024>>)

      .def("v_x_flux", &Discretization::v_x_flux<Mesh<10, 10>>)
      .def("v_x_flux", &Discretization::v_x_flux<Mesh<20, 20>>)
      .def("v_x_flux", &Discretization::v_x_flux<Mesh<40, 40>>)
      .def("v_x_flux", &Discretization::v_x_flux<Mesh<80, 80>>)
      .def("v_x_flux", &Discretization::v_x_flux<Mesh<160, 160>>)
      .def("v_x_flux", &Discretization::v_x_flux<Mesh<1024, 1024>>)

      .def("v_y_flux", &Discretization::v_y_flux<Mesh<10, 10>>)
      .def("v_y_flux", &Discretization::v_y_flux<Mesh<20, 20>>)
      .def("v_y_flux", &Discretization::v_y_flux<Mesh<40, 40>>)
      .def("v_y_flux", &Discretization::v_y_flux<Mesh<80, 80>>)
      .def("v_y_flux", &Discretization::v_y_flux<Mesh<160, 160>>)
      .def("v_y_flux", &Discretization::v_y_flux<Mesh<1024, 1024>>)

      .def("du_x_flux", &Discretization::du_x_flux<Mesh<10, 10>>)
      .def("du_x_flux", &Discretization::du_x_flux<Mesh<20, 20>>)
      .def("du_x_flux", &Discretization::du_x_flux<Mesh<40, 40>>)
      .def("du_x_flux", &Discretization::du_x_flux<Mesh<80, 80>>)
      .def("du_x_flux", &Discretization::du_x_flux<Mesh<160, 160>>)
      .def("du_x_flux", &Discretization::du_x_flux<Mesh<1024, 1024>>)

      .def("du_y_flux", &Discretization::du_y_flux<Mesh<10, 10>>)
      .def("du_y_flux", &Discretization::du_y_flux<Mesh<20, 20>>)
      .def("du_y_flux", &Discretization::du_y_flux<Mesh<40, 40>>)
      .def("du_y_flux", &Discretization::du_y_flux<Mesh<80, 80>>)
      .def("du_y_flux", &Discretization::du_y_flux<Mesh<160, 160>>)
      .def("du_y_flux", &Discretization::du_y_flux<Mesh<1024, 1024>>)

      .def("dv_x_flux", &Discretization::dv_x_flux<Mesh<10, 10>>)
      .def("dv_x_flux", &Discretization::dv_x_flux<Mesh<20, 20>>)
      .def("dv_x_flux", &Discretization::dv_x_flux<Mesh<40, 40>>)
      .def("dv_x_flux", &Discretization::dv_x_flux<Mesh<80, 80>>)
      .def("dv_x_flux", &Discretization::dv_x_flux<Mesh<160, 160>>)
      .def("dv_x_flux", &Discretization::dv_x_flux<Mesh<1024, 1024>>)

      .def("dv_y_flux", &Discretization::dv_y_flux<Mesh<10, 10>>)
      .def("dv_y_flux", &Discretization::dv_y_flux<Mesh<20, 20>>)
      .def("dv_y_flux", &Discretization::dv_y_flux<Mesh<40, 40>>)
      .def("dv_y_flux", &Discretization::dv_y_flux<Mesh<80, 80>>)
      .def("dv_y_flux", &Discretization::dv_y_flux<Mesh<160, 160>>)
      .def("dv_y_flux", &Discretization::dv_y_flux<Mesh<1024, 1024>>);

  using Assembly = INSAssembly<Discretization>;
  py::class_<Assembly, Discretization>(module, "INSAssembly")
      .def(py::init<const BConds_Part1 &>(), py::arg("boundaries"))
      // .def("boundaries",
      //      [](Assembly &self) {
      //        printf("self: %p\n", &self);
      //        return self.boundaries();
      //      })
      .def("boundaries", &Assembly::boundaries)
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

      .def("jacobian_x_p1", &Assembly::jacobian_x_p1<Mesh<10, 10>>)
      .def("jacobian_x_p1", &Assembly::jacobian_x_p1<Mesh<20, 20>>)
      .def("jacobian_x_p1", &Assembly::jacobian_x_p1<Mesh<40, 40>>)
      .def("jacobian_x_p1", &Assembly::jacobian_x_p1<Mesh<80, 80>>)
      .def("jacobian_x_p1", &Assembly::jacobian_x_p1<Mesh<160, 160>>)
      .def("jacobian_x_p1", &Assembly::jacobian_x_p1<Mesh<1024, 1024>>)

      .def("jacobian_x_0", &Assembly::jacobian_x_0<Mesh<10, 10>>)
      .def("jacobian_x_0", &Assembly::jacobian_x_0<Mesh<20, 20>>)
      .def("jacobian_x_0", &Assembly::jacobian_x_0<Mesh<40, 40>>)
      .def("jacobian_x_0", &Assembly::jacobian_x_0<Mesh<80, 80>>)
      .def("jacobian_x_0", &Assembly::jacobian_x_0<Mesh<160, 160>>)
      .def("jacobian_x_0", &Assembly::jacobian_x_0<Mesh<1024, 1024>>)

      .def("jacobian_y_p1", &Assembly::jacobian_y_p1<Mesh<10, 10>>)
      .def("jacobian_y_p1", &Assembly::jacobian_y_p1<Mesh<20, 20>>)
      .def("jacobian_y_p1", &Assembly::jacobian_y_p1<Mesh<40, 40>>)
      .def("jacobian_y_p1", &Assembly::jacobian_y_p1<Mesh<80, 80>>)
      .def("jacobian_y_p1", &Assembly::jacobian_y_p1<Mesh<160, 160>>)
      .def("jacobian_y_p1", &Assembly::jacobian_y_p1<Mesh<1024, 1024>>)

      .def("jacobian_y_0", &Assembly::jacobian_y_0<Mesh<10, 10>>)
      .def("jacobian_y_0", &Assembly::jacobian_y_0<Mesh<20, 20>>)
      .def("jacobian_y_0", &Assembly::jacobian_y_0<Mesh<40, 40>>)
      .def("jacobian_y_0", &Assembly::jacobian_y_0<Mesh<80, 80>>)
      .def("jacobian_y_0", &Assembly::jacobian_y_0<Mesh<160, 160>>)
      .def("jacobian_y_0", &Assembly::jacobian_y_0<Mesh<1024, 1024>>);
}
