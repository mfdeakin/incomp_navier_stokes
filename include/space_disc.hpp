
#ifndef _SPACE_DISC_HPP_
#define _SPACE_DISC_HPP_

#include "constants.hpp"

#include <cmath>
#include <functional>

// Use the curiously repeated template parameter to swap out the order of the
// discretization in our assembly
template <typename _SpaceDisc>
class [[nodiscard]] EnergyAssembly : public _SpaceDisc {
 public:
  // u = u_0 y \sin(\pi x)
  // v = v_0 x \cos(\pi y)
  using SpaceDisc = _SpaceDisc;

  template <typename MeshT>
  [[nodiscard]] real flux_integral(const MeshT &mesh, int i, int j,
                                   const real time) const noexcept {
    const real x_deriv = (this->uT_x_flux(mesh, i, j, time) -
                          this->uT_x_flux(mesh, i - 1, j, time)) /
                         mesh.dx();
    const real y_deriv = (this->vT_y_flux(mesh, i, j, time) -
                          this->vT_y_flux(mesh, i, j - 1, time)) /
                         mesh.dy();

    return (-x_deriv - y_deriv) +
           _diffuse_coeff * nabla2_T_flux_integral(mesh, i, j, time);
  }

  template <typename MeshT>
  [[nodiscard]] real nabla2_T_flux_integral(const MeshT &mesh, int i, int j,
                                            const real time) const noexcept {
    const real x2_deriv = (this->dx_flux(mesh, i, j, time) -
                           this->dx_flux(mesh, i - 1, j, time)) /
                          mesh.dx();
    const real y2_deriv = (this->dy_flux(mesh, i, j, time) -
                           this->dy_flux(mesh, i, j - 1, time)) /
                          mesh.dy();
    return (x2_deriv + y2_deriv) / (reynolds * prandtl);
  }

  template <typename MeshT>
  void flux_assembly(const MeshT &initial, const MeshT &current, MeshT &next,
                     const real time, const real dt) const noexcept {
    for(int i = 0; i < initial.x_dim(); i++) {
      for(int j = 0; j < initial.y_dim(); j++) {
        next.Temp(i, j) =
            initial.Temp(i, j) + dt * (flux_integral(current, i, j, time) +
                                       this->source_fd(current, i, j, time));
      }
    }
  }

  // Setting the coefficient for diffusion doesn't correspond to any physical
  // process, so default it to it's physical value
  constexpr EnergyAssembly(const real T_0, const real u_0, const real v_0,
                           const real diffusion = 1.0) noexcept
      : SpaceDisc(T_0, u_0, v_0), _diffuse_coeff(diffusion) {}

 protected:
  const real _diffuse_coeff;
};

// TODO: Split up the boundary condition parts from the second order centered
// calculations. This will save ~400 lines of code and reduce copy-pasta errors
class [[nodiscard]] SecondOrderCentered_Part7 {
 public:
  // Terms for implicit euler; p1 is dim +1, 0 is just dim, m1 is dim - 1
  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_p1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(i < mesh.x_dim() - 1) {
      return mesh.u_vel(i + 1, j) / (2.0 * mesh.dx()) -
             1.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_0(const MeshT &mesh, int i, int j)
      const noexcept {
    return 2.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_m1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(i > 0) {
      return -mesh.u_vel(i - 1, j) / (2.0 * mesh.dx()) -
             1.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_p1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(j < mesh.y_dim() - 1) {
      return mesh.v_vel(i, j + 1) / (2.0 * mesh.dy()) -
             1.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_0(const MeshT &mesh, int i, int j)
      const noexcept {
    return 2.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_m1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(j > 0) {
      return -mesh.v_vel(i, j - 1) / (2.0 * mesh.dy()) -
             1.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
    } else {
      return 0.0;
    }
  }

  // Centered FV approximation to (u T)_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real uT_x_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    if(i == mesh.x_dim() - 1) {
      const real y = mesh.y_median(j);
      return boundary_x_1(y, time) * u(x_max(), y);
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      return boundary_x_0(y, time) * u(x_min(), y);
    } else {
      return (mesh.Temp(i, j) * mesh.u_vel(i, j) +
              mesh.Temp(i + 1, j) * mesh.u_vel(i + 1, j)) /
             2.0;
    }
  }

  // Centered FV approximation to T_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real vT_y_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    if(j == mesh.y_dim() - 1) {
      const real x = mesh.x_median(i);
      return boundary_y_1(x, time) * v(x, y_max());
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      return boundary_y_0(x, time) * v(x, y_min());
    } else {
      return (mesh.Temp(i, j) * mesh.v_vel(i, j) +
              mesh.Temp(i, j + 1) * mesh.v_vel(i, j + 1)) /
             2.0;
    }
  }

  // Centered FV approximation to dT/dx_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dx_flux(const MeshT &mesh, const int i,
                                       const int j, const real time)
      const noexcept {
    if(i == mesh.x_dim() - 1) {
      const real y       = mesh.y_median(j);
      const real T_right = 2.0 * boundary_x_1(y, time) - mesh.Temp(i, j);
      return (T_right - mesh.Temp(i, j)) / mesh.dx();
    } else if(i == -1) {
      const real y      = mesh.y_median(j);
      const real T_left = 2.0 * boundary_x_0(y, time) - mesh.Temp(i + 1, j);
      return (mesh.Temp(i + 1, j) - T_left) / mesh.dx();
    } else {
      return (mesh.Temp(i + 1, j) - mesh.Temp(i, j)) / mesh.dx();
    }
  }

  // Centered FV approximation to dT/dy_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real dy_flux(const MeshT &mesh, const int i,
                                       const int j, const real time)
      const noexcept {
    if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real T_above = 2.0 * boundary_y_1(x, time) - mesh.Temp(i, j);
      return (T_above - mesh.Temp(i, j)) / mesh.dy();
    } else if(j == -1) {
      const real x       = mesh.x_median(i);
      const real T_below = 2.0 * boundary_y_0(x, time) - mesh.Temp(i, j + 1);
      return (mesh.Temp(i, j + 1) - T_below) / mesh.dy();
    } else {
      return (mesh.Temp(i, j + 1) - mesh.Temp(i, j)) / mesh.dy();
    }
  }

  // Uses the finite difference (FD) approximations to the velocity derivatives
  // to approximate the source term
  template <typename MeshT>
  [[nodiscard]] constexpr real source_fd(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    const real u_dx = du_dx_fd(mesh, i, j);
    const real v_dy = dv_dy_fd(mesh, i, j);
    const real u_dy = du_dy_fd(mesh, i, j);
    const real v_dx = dv_dx_fd(mesh, i, j);

    const real cross_term = u_dy + v_dx;
    return eckert / reynolds *
           (2.0 * (u_dx * u_dx + v_dy * v_dy) + cross_term * cross_term);
  }

  // Centered FD approximation to du/dx_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real du_dx_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(i == 0) {
      const real x_left = mesh.x_median(i - 1);
      const real y      = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (mesh.u_vel(i + 1, j) - u(x_left, y)) / (2.0 * mesh.dx());
    } else if(i == mesh.x_dim() - 1) {
      const real x_right = mesh.x_median(i + 1);
      const real y       = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (u(x_right, y) - mesh.u_vel(i - 1, j)) / (2.0 * mesh.dx());
    } else {
      return (mesh.u_vel(i + 1, j) - mesh.u_vel(i - 1, j)) / (2.0 * mesh.dx());
    }
  }

  // Centered FD approximation to du/dy_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real du_dy_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(j == 0) {
      const real x       = mesh.x_median(i);
      const real y_below = mesh.y_median(j - 1);
      // Use our exact solution to u outside of the boundaries
      return (mesh.u_vel(i, j + 1) - u(x, y_below)) / (2.0 * mesh.dy());
    } else if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real y_above = mesh.y_median(j + 1);
      // Use our exact solution to u outside of the boundaries
      return (u(x, y_above) - mesh.u_vel(i, j - 1)) / (2.0 * mesh.dy());
    } else {
      return (mesh.u_vel(i, j + 1) - mesh.u_vel(i, j - 1)) / (2.0 * mesh.dy());
    }
  }

  // Centered FD approximation to dv/dx_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dv_dx_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(i == 0) {
      const real x_left = mesh.x_median(i - 1);
      const real y      = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (mesh.v_vel(i + 1, j) - v(x_left, y)) / (2.0 * mesh.dx());
    } else if(i == mesh.x_dim() - 1) {
      const real x_right = mesh.x_median(i + 1);
      const real y       = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (v(x_right, y) - mesh.v_vel(i - 1, j)) / (2.0 * mesh.dx());
    } else {
      return (mesh.v_vel(i + 1, j) - mesh.v_vel(i - 1, j)) / (2.0 * mesh.dx());
    }
  }

  // Centered FD approximation to dv/dy_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dv_dy_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(j == 0) {
      const real x       = mesh.x_median(i);
      const real y_below = mesh.y_median(j - 1);
      // Use our exact solution to u outside of the boundaries
      return (mesh.v_vel(i, j + 1) - v(x, y_below)) / (2.0 * mesh.dy());
    } else if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real y_above = mesh.y_median(j + 1);
      // Use our exact solution to u outside of the boundaries
      return (v(x, y_above) - mesh.v_vel(i, j - 1)) / (2.0 * mesh.dy());
    } else {
      return (mesh.v_vel(i, j + 1) - mesh.v_vel(i, j - 1)) / (2.0 * mesh.dy());
    }
  }

  [[nodiscard]] static constexpr real x_min() noexcept { return 0.0; }
  [[nodiscard]] static constexpr real y_min() noexcept { return 0.0; }
  [[nodiscard]] static constexpr real x_max() noexcept { return 1.0; }
  [[nodiscard]] static constexpr real y_max() noexcept { return 1.0; }

  // T = T_0 \cos(\pi x) \sin(\pi y)
  [[nodiscard]] constexpr real T_0() const noexcept { return _t_0; }

  [[nodiscard]] constexpr real u_0() const noexcept { return _u_0; }

  [[nodiscard]] constexpr real v_0() const noexcept { return _v_0; }

  // The rest of the methods are just related to the analytic solution
  // These are needed at the minimum to implement the boundary conditions,
  // and are also used for computing the error norms
  [[nodiscard]] real flux_int_solution(const real x, const real y)
      const noexcept {
    const real vel_nabla_t_fi = flux_int_vel_nabla_T_sol(x, y);
    const real diffusion_fi   = flux_int_nabla2_T_sol(x, y);
    return vel_nabla_t_fi + diffusion_fi;
  }

  [[nodiscard]] real flux_int_vel_nabla_T_sol(const real x, const real y)
      const noexcept {
    const real u_dt_dy_fi =
        pi * u_0() * std::cos(2.0 * pi * x) * y * std::sin(pi * y);
    const real v_dt_dy_fi =
        pi * v_0() * x * std::cos(pi * x) * std::cos(2.0 * pi * y);
    return -T_0() * (u_dt_dy_fi + v_dt_dy_fi);
  }

  [[nodiscard]] real flux_int_nabla2_T_sol(const real x, const real y)
      const noexcept {
    return -T_0() * (2.0 * pi * pi / (reynolds * prandtl)) * std::cos(pi * x) *
           std::sin(pi * y);
  }

  // Use the exact solutions to implement the boundary conditions and also check
  // that the flux integral is correct
  [[nodiscard]] real solution(const real x, const real y, const real time)
      const noexcept {
    if(time == 0.0) {
      return y;
    } else if(x == x_min()) {
      const real y_term_1 = 1.0 - 2.0 * y;
      const real y_term_2 = y_term_1 * y_term_1;
      return y + 0.75 * prandtl * eckert * u_0() * u_0() *
                     (1.0 - y_term_2 * y_term_2);
    } else if(x == x_max()) {
      const real y_term_1 = 1.0 - 2.0 * y;
      const real y_term_2 = y_term_1 * y_term_1;
      return y + 0.75 * prandtl * eckert * u_0() * u_0() *
                     (1.0 - y_term_2 * y_term_2);
    } else if(y == y_min()) {
      return 0.0;
    } else if(y == y_max()) {
      return 1.0;
    } else {
      return std::numeric_limits<real>::quiet_NaN();
    }
  }

  [[nodiscard]] real source_sol(const real x, const real y, const real time)
      const noexcept {
    const real u_dx = u_0() * pi * std::cos(pi * x) * y;
    const real v_dy = v_0() * pi * std::sin(pi * y) * x;
    const real u_dy = u_0() * std::sin(pi * x);
    const real v_dx = v_0() * std::cos(pi * y);

    const real cross_term = u_dy + v_dx;
    return eckert / reynolds *
           (2.0 * (u_dx * u_dx + v_dy * v_dy) + cross_term * cross_term);
  }

  [[nodiscard]] real u(const real x, const real y) const noexcept {
    return u_0() * 6.0 * y * (1.0 - y);
  }

  [[nodiscard]] real v(const real x, const real y) const noexcept {
    return 0.0;
  }

  // To make filling the mesh with the initial solution easier
  [[nodiscard]] std::function<std::tuple<real, real, real>(real, real)>
  initial_solution_tuple() const noexcept {
    return [=](const real x, const real y) {
      return std::tuple<real, real, real>(solution(x, y, 0.0), u(x, y),
                                          v(x, y));
    };
  }

  [[nodiscard]] real solution_dx(const real x, const real y, const real time)
      const noexcept {
    return -pi * T_0() * std::sin(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy(const real x, const real y, const real time)
      const noexcept {
    return pi * T_0() * std::cos(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real solution_dx2(const real x, const real y, const real time)
      const noexcept {
    return -pi * pi * T_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy2(const real x, const real y, const real time)
      const noexcept {
    return solution_dx2(x, y, time);
  }

  [[nodiscard]] real boundary_x_0(const real y, const real time)
      const noexcept {
    return solution(x_min(), y, time);
  }

  [[nodiscard]] real boundary_x_1(const real y, const real time)
      const noexcept {
    return solution(x_max(), y, time);
  }

  [[nodiscard]] real boundary_y_0(const real x, const real time)
      const noexcept {
    return solution(x, y_min(), time);
  }

  [[nodiscard]] real boundary_y_1(const real x, const real time)
      const noexcept {
    return solution(x, y_max(), time);
  }

  constexpr SecondOrderCentered_Part7(const real T_0, const real u_0,
                                      const real v_0) noexcept
      : _t_0(T_0), _u_0(u_0), _v_0(v_0) {}

  constexpr SecondOrderCentered_Part7(
      const SecondOrderCentered_Part7 &src) noexcept
      : SecondOrderCentered_Part7(src.T_0(), src.u_0(), src.v_0()) {}

 protected:
  const real _t_0;
  const real _u_0, _v_0;
};

class [[nodiscard]] SecondOrderCentered_Part5 {
 public:
  // Terms for implicit euler
  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_p1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(i < mesh.x_dim() - 1) {
      return mesh.u_vel(i + 1, j) / (2.0 * mesh.dx()) -
             1.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_0(const MeshT &mesh, int i, int j)
      const noexcept {
    return 2.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_m1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(i > 0) {
      return -mesh.u_vel(i - 1, j) / (2.0 * mesh.dx()) -
             1.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_p1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(j < mesh.y_dim() - 1) {
      return mesh.v_vel(i, j + 1) / (2.0 * mesh.dy()) -
             1.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_0(const MeshT &mesh, int i, int j)
      const noexcept {
    return 2.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_m1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(j > 0) {
      return -mesh.v_vel(i, j - 1) / (2.0 * mesh.dy()) -
             1.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
    } else {
      return 0.0;
    }
  }

  // Centered FV approximation to (u T)_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real uT_x_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    if(i == mesh.x_dim() - 1) {
      const real y = mesh.y_median(j);
      return boundary_x_1(y, time) * u(x_max(), y);
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      return boundary_x_0(y, time) * u(x_min(), y);
    } else {
      return (mesh.Temp(i, j) * mesh.u_vel(i, j) +
              mesh.Temp(i + 1, j) * mesh.u_vel(i + 1, j)) /
             2.0;
    }
  }

  // Centered FV approximation to T_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real vT_y_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    if(j == mesh.y_dim() - 1) {
      const real x = mesh.x_median(i);
      return boundary_y_1(x, time) * v(x, y_max());
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      return boundary_y_0(x, time) * v(x, y_min());
    } else {
      return (mesh.Temp(i, j) * mesh.v_vel(i, j) +
              mesh.Temp(i, j + 1) * mesh.v_vel(i, j + 1)) /
             2.0;
    }
  }

  // Centered FV approximation to dT/dx_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dx_flux(const MeshT &mesh, const int i,
                                       const int j, const real time)
      const noexcept {
    if(i == mesh.x_dim() - 1) {
      const real y       = mesh.y_median(j);
      const real T_right = 2.0 * boundary_x_1(y, time) - mesh.Temp(i, j);
      return (T_right - mesh.Temp(i, j)) / mesh.dx();
    } else if(i == -1) {
      const real y      = mesh.y_median(j);
      const real T_left = 2.0 * boundary_x_0(y, time) - mesh.Temp(i + 1, j);
      return (mesh.Temp(i + 1, j) - T_left) / mesh.dx();
    } else {
      return (mesh.Temp(i + 1, j) - mesh.Temp(i, j)) / mesh.dx();
    }
  }

  // Centered FV approximation to dT/dy_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real dy_flux(const MeshT &mesh, const int i,
                                       const int j, const real time)
      const noexcept {
    if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real T_above = 2.0 * boundary_y_1(x, time) - mesh.Temp(i, j);
      return (T_above - mesh.Temp(i, j)) / mesh.dy();
    } else if(j == -1) {
      const real x       = mesh.x_median(i);
      const real T_below = 2.0 * boundary_y_0(x, time) - mesh.Temp(i, j + 1);
      return (mesh.Temp(i, j + 1) - T_below) / mesh.dy();
    } else {
      return (mesh.Temp(i, j + 1) - mesh.Temp(i, j)) / mesh.dy();
    }
  }

  // Uses the finite difference (FD) approximations to the velocity derivatives
  // to approximate the source term
  template <typename MeshT>
  [[nodiscard]] constexpr real source_fd(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    const real u_dx = du_dx_fd(mesh, i, j);
    const real v_dy = dv_dy_fd(mesh, i, j);
    const real u_dy = du_dy_fd(mesh, i, j);
    const real v_dx = dv_dx_fd(mesh, i, j);

    const real cross_term = u_dy + v_dx;
    return eckert / reynolds *
           (2.0 * (u_dx * u_dx + v_dy * v_dy) + cross_term * cross_term);
  }

  // Centered FD approximation to du/dx_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real du_dx_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(i == 0) {
      const real x_left = mesh.x_median(i - 1);
      const real y      = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (mesh.u_vel(i + 1, j) - u(x_left, y)) / (2.0 * mesh.dx());
    } else if(i == mesh.x_dim() - 1) {
      const real x_right = mesh.x_median(i + 1);
      const real y       = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (u(x_right, y) - mesh.u_vel(i - 1, j)) / (2.0 * mesh.dx());
    } else {
      return (mesh.u_vel(i + 1, j) - mesh.u_vel(i - 1, j)) / (2.0 * mesh.dx());
    }
  }

  // Centered FD approximation to du/dy_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real du_dy_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(j == 0) {
      const real x       = mesh.x_median(i);
      const real y_below = mesh.y_median(j - 1);
      // Use our exact solution to u outside of the boundaries
      return (mesh.u_vel(i, j + 1) - u(x, y_below)) / (2.0 * mesh.dy());
    } else if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real y_above = mesh.y_median(j + 1);
      // Use our exact solution to u outside of the boundaries
      return (u(x, y_above) - mesh.u_vel(i, j - 1)) / (2.0 * mesh.dy());
    } else {
      return (mesh.u_vel(i, j + 1) - mesh.u_vel(i, j - 1)) / (2.0 * mesh.dy());
    }
  }

  // Centered FD approximation to dv/dx_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dv_dx_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(i == 0) {
      const real x_left = mesh.x_median(i - 1);
      const real y      = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (mesh.v_vel(i + 1, j) - v(x_left, y)) / (2.0 * mesh.dx());
    } else if(i == mesh.x_dim() - 1) {
      const real x_right = mesh.x_median(i + 1);
      const real y       = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (v(x_right, y) - mesh.v_vel(i - 1, j)) / (2.0 * mesh.dx());
    } else {
      return (mesh.v_vel(i + 1, j) - mesh.v_vel(i - 1, j)) / (2.0 * mesh.dx());
    }
  }

  // Centered FD approximation to dv/dy_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dv_dy_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(j == 0) {
      const real x       = mesh.x_median(i);
      const real y_below = mesh.y_median(j - 1);
      // Use our exact solution to u outside of the boundaries
      return (mesh.v_vel(i, j + 1) - v(x, y_below)) / (2.0 * mesh.dy());
    } else if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real y_above = mesh.y_median(j + 1);
      // Use our exact solution to u outside of the boundaries
      return (v(x, y_above) - mesh.v_vel(i, j - 1)) / (2.0 * mesh.dy());
    } else {
      return (mesh.v_vel(i, j + 1) - mesh.v_vel(i, j - 1)) / (2.0 * mesh.dy());
    }
  }

  [[nodiscard]] static constexpr real x_min() noexcept { return 0.0; }
  [[nodiscard]] static constexpr real y_min() noexcept { return 0.0; }
  [[nodiscard]] static constexpr real x_max() noexcept { return 1.0; }
  [[nodiscard]] static constexpr real y_max() noexcept { return 1.0; }

  // T = T_0 \cos(\pi x) \sin(\pi y)
  [[nodiscard]] constexpr real T_0() const noexcept { return _t_0; }

  [[nodiscard]] constexpr real u_0() const noexcept { return _u_0; }

  [[nodiscard]] constexpr real v_0() const noexcept { return _v_0; }

  // The rest of the methods are just related to the analytic solution
  // These are needed at the minimum to implement the boundary conditions,
  // and are also used for computing the error norms
  [[nodiscard]] real flux_int_solution(const real x, const real y)
      const noexcept {
    const real vel_nabla_t_fi = flux_int_vel_nabla_T_sol(x, y);
    const real diffusion_fi   = flux_int_nabla2_T_sol(x, y);
    return vel_nabla_t_fi + diffusion_fi;
  }

  [[nodiscard]] real flux_int_vel_nabla_T_sol(const real x, const real y)
      const noexcept {
    const real u_dt_dy_fi =
        pi * u_0() * std::cos(2.0 * pi * x) * y * std::sin(pi * y);
    const real v_dt_dy_fi =
        pi * v_0() * x * std::cos(pi * x) * std::cos(2.0 * pi * y);
    return -T_0() * (u_dt_dy_fi + v_dt_dy_fi);
  }

  [[nodiscard]] real flux_int_nabla2_T_sol(const real x, const real y)
      const noexcept {
    return -T_0() * (2.0 * pi * pi / (reynolds * prandtl)) * std::cos(pi * x) *
           std::sin(pi * y);
  }

  // Use the exact solutions to implement the boundary conditions and also check
  // that the flux integral is correct
  [[nodiscard]] real solution(const real x, const real y, const real time)
      const noexcept {
    if(time == 0.0) {
      return y;
    } else if(x == x_min()) {
      const real y_term_1 = 1.0 - 2.0 * y;
      const real y_term_2 = y_term_1 * y_term_1;
      return y + 0.75 * prandtl * eckert * u_0() * u_0() *
                     (1.0 - y_term_2 * y_term_2);
    } else if(x == x_max()) {
      const real y_term_1 = 1.0 - 2.0 * y;
      const real y_term_2 = y_term_1 * y_term_1;
      return y + 0.75 * prandtl * eckert * u_0() * u_0() *
                     (1.0 - y_term_2 * y_term_2);
    } else if(y == y_min()) {
      return 0.0;
    } else if(y == y_max()) {
      return 1.0;
    } else {
      return std::numeric_limits<real>::quiet_NaN();
    }
  }

  [[nodiscard]] real source_sol(const real x, const real y, const real time)
      const noexcept {
    const real u_dx = u_0() * pi * std::cos(pi * x) * y;
    const real v_dy = v_0() * pi * std::sin(pi * y) * x;
    const real u_dy = u_0() * std::sin(pi * x);
    const real v_dx = v_0() * std::cos(pi * y);

    const real cross_term = u_dy + v_dx;
    return eckert / reynolds *
           (2.0 * (u_dx * u_dx + v_dy * v_dy) + cross_term * cross_term);
  }

  [[nodiscard]] real u(const real x, const real y) const noexcept {
    return u_0() * 6.0 * y * (1.0 - y);
  }

  [[nodiscard]] real v(const real x, const real y) const noexcept {
    return 0.0;
  }

  // To make filling the mesh with the initial solution easier
  [[nodiscard]] std::function<std::tuple<real, real, real>(real, real)>
  initial_solution_tuple() const noexcept {
    return [=](const real x, const real y) {
      return std::tuple<real, real, real>(solution(x, y, 0.0), u(x, y),
                                          v(x, y));
    };
  }

  [[nodiscard]] real solution_dx(const real x, const real y, const real time)
      const noexcept {
    return -pi * T_0() * std::sin(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy(const real x, const real y, const real time)
      const noexcept {
    return pi * T_0() * std::cos(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real solution_dx2(const real x, const real y, const real time)
      const noexcept {
    return -pi * pi * T_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy2(const real x, const real y, const real time)
      const noexcept {
    return solution_dx2(x, y, time);
  }

  [[nodiscard]] real boundary_x_0(const real y, const real time)
      const noexcept {
    return solution(x_min(), y, time);
  }

  [[nodiscard]] real boundary_x_1(const real y, const real time)
      const noexcept {
    return solution(x_max(), y, time);
  }

  [[nodiscard]] real boundary_y_0(const real x, const real time)
      const noexcept {
    return solution(x, y_min(), time);
  }

  [[nodiscard]] real boundary_y_1(const real x, const real time)
      const noexcept {
    return solution(x, y_max(), time);
  }

  constexpr SecondOrderCentered_Part5(const real T_0, const real u_0,
                                      const real v_0) noexcept
      : _t_0(T_0), _u_0(u_0), _v_0(v_0) {}

  constexpr SecondOrderCentered_Part5(
      const SecondOrderCentered_Part5 &src) noexcept
      : SecondOrderCentered_Part5(src.T_0(), src.u_0(), src.v_0()) {}

 protected:
  const real _t_0;
  const real _u_0, _v_0;
};

class [[nodiscard]] SecondOrderCentered_Part1 {
 public:
  // Terms for implicit euler
  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_p1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(i < mesh.x_dim() - 1) {
      return mesh.u_vel(i + 1, j) / (2.0 * mesh.dx()) -
             1.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_0(const MeshT &mesh, int i, int j)
      const noexcept {
    return 2.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dx_m1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(i > 0) {
      return -mesh.u_vel(i - 1, j) / (2.0 * mesh.dx()) -
             1.0 / (reynolds * prandtl * mesh.dx() * mesh.dx());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_p1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(j < mesh.y_dim() - 1) {
      return mesh.v_vel(i, j + 1) / (2.0 * mesh.dy()) -
             1.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
    } else {
      return 0.0;
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_0(const MeshT &mesh, int i, int j)
      const noexcept {
    return 2.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real Dy_m1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(j > 0) {
      return -mesh.v_vel(i, j - 1) / (2.0 * mesh.dy()) -
             1.0 / (reynolds * prandtl * mesh.dy() * mesh.dy());
    } else {
      return 0.0;
    }
  }

  // Centered FV approximation to (u T)_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real uT_x_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    if(i == mesh.x_dim() - 1) {
      const real y = mesh.y_median(j);
      return boundary_x_1(y, time) * u(x_max(), y);
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      return boundary_x_0(y, time) * u(x_min(), y);
    } else {
      return (mesh.Temp(i, j) * mesh.u_vel(i, j) +
              mesh.Temp(i + 1, j) * mesh.u_vel(i + 1, j)) /
             2.0;
    }
  }

  // Centered FV approximation to T_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real vT_y_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    if(j == mesh.y_dim() - 1) {
      const real x = mesh.x_median(i);
      return boundary_y_1(x, time) * v(x, y_max());
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      return boundary_y_0(x, time) * v(x, y_min());
    } else {
      return (mesh.Temp(i, j) * mesh.v_vel(i, j) +
              mesh.Temp(i, j + 1) * mesh.v_vel(i, j + 1)) /
             2.0;
    }
  }

  // Centered FV approximation to dT/dx_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dx_flux(const MeshT &mesh, const int i,
                                       const int j, const real time)
      const noexcept {
    if(i == mesh.x_dim() - 1) {
      const real y       = mesh.y_median(j);
      const real T_right = 2.0 * boundary_x_1(y, time) - mesh.Temp(i, j);
      return (T_right - mesh.Temp(i, j)) / mesh.dx();
    } else if(i == -1) {
      const real y      = mesh.y_median(j);
      const real T_left = 2.0 * boundary_x_0(y, time) - mesh.Temp(i + 1, j);
      return (mesh.Temp(i + 1, j) - T_left) / mesh.dx();
    } else {
      return (mesh.Temp(i + 1, j) - mesh.Temp(i, j)) / mesh.dx();
    }
  }

  // Centered FV approximation to dT/dy_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real dy_flux(const MeshT &mesh, const int i,
                                       const int j, const real time)
      const noexcept {
    if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real T_above = 2.0 * boundary_y_1(x, time) - mesh.Temp(i, j);
      return (T_above - mesh.Temp(i, j)) / mesh.dy();
    } else if(j == -1) {
      const real x       = mesh.x_median(i);
      const real T_below = 2.0 * boundary_y_0(x, time) - mesh.Temp(i, j + 1);
      return (mesh.Temp(i, j + 1) - T_below) / mesh.dy();
    } else {
      return (mesh.Temp(i, j + 1) - mesh.Temp(i, j)) / mesh.dy();
    }
  }

  // Uses the finite difference (FD) approximations to the velocity derivatives
  // to approximate the source term
  template <typename MeshT>
  [[nodiscard]] constexpr real source_fd(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    const real u_dx = du_dx_fd(mesh, i, j);
    const real v_dy = dv_dy_fd(mesh, i, j);
    const real u_dy = du_dy_fd(mesh, i, j);
    const real v_dx = dv_dx_fd(mesh, i, j);

    const real cross_term = u_dy + v_dx;
    return eckert / reynolds *
           (2.0 * (u_dx * u_dx + v_dy * v_dy) + cross_term * cross_term);
  }

  // Centered FD approximation to du/dx_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real du_dx_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(i == 0) {
      const real x_left = mesh.x_median(i - 1);
      const real y      = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (mesh.u_vel(i + 1, j) - u(x_left, y)) / (2.0 * mesh.dx());
    } else if(i == mesh.x_dim() - 1) {
      const real x_right = mesh.x_median(i + 1);
      const real y       = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (u(x_right, y) - mesh.u_vel(i - 1, j)) / (2.0 * mesh.dx());
    } else {
      return (mesh.u_vel(i + 1, j) - mesh.u_vel(i - 1, j)) / (2.0 * mesh.dx());
    }
  }

  // Centered FD approximation to du/dy_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real du_dy_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(j == 0) {
      const real x       = mesh.x_median(i);
      const real y_below = mesh.y_median(j - 1);
      // Use our exact solution to u outside of the boundaries
      return (mesh.u_vel(i, j + 1) - u(x, y_below)) / (2.0 * mesh.dy());
    } else if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real y_above = mesh.y_median(j + 1);
      // Use our exact solution to u outside of the boundaries
      return (u(x, y_above) - mesh.u_vel(i, j - 1)) / (2.0 * mesh.dy());
    } else {
      return (mesh.u_vel(i, j + 1) - mesh.u_vel(i, j - 1)) / (2.0 * mesh.dy());
    }
  }

  // Centered FD approximation to dv/dx_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dv_dx_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(i == 0) {
      const real x_left = mesh.x_median(i - 1);
      const real y      = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (mesh.v_vel(i + 1, j) - v(x_left, y)) / (2.0 * mesh.dx());
    } else if(i == mesh.x_dim() - 1) {
      const real x_right = mesh.x_median(i + 1);
      const real y       = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (v(x_right, y) - mesh.v_vel(i - 1, j)) / (2.0 * mesh.dx());
    } else {
      return (mesh.v_vel(i + 1, j) - mesh.v_vel(i - 1, j)) / (2.0 * mesh.dx());
    }
  }

  // Centered FD approximation to dv/dy_{i, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dv_dy_fd(const MeshT &mesh, const int i,
                                        const int j) const noexcept {
    if(j == 0) {
      const real x       = mesh.x_median(i);
      const real y_below = mesh.y_median(j - 1);
      // Use our exact solution to u outside of the boundaries
      return (mesh.v_vel(i, j + 1) - v(x, y_below)) / (2.0 * mesh.dy());
    } else if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real y_above = mesh.y_median(j + 1);
      // Use our exact solution to u outside of the boundaries
      return (v(x, y_above) - mesh.v_vel(i, j - 1)) / (2.0 * mesh.dy());
    } else {
      return (mesh.v_vel(i, j + 1) - mesh.v_vel(i, j - 1)) / (2.0 * mesh.dy());
    }
  }

  [[nodiscard]] static constexpr real x_min() noexcept { return 0.0; }
  [[nodiscard]] static constexpr real y_min() noexcept { return 0.0; }
  [[nodiscard]] static constexpr real x_max() noexcept { return 1.0; }
  [[nodiscard]] static constexpr real y_max() noexcept { return 1.0; }

  // T = T_0 \cos(\pi x) \sin(\pi y)
  [[nodiscard]] constexpr real T_0() const noexcept { return _t_0; }

  [[nodiscard]] constexpr real u_0() const noexcept { return _u_0; }

  [[nodiscard]] constexpr real v_0() const noexcept { return _v_0; }

  // The rest of the methods are just related to the analytic solution
  // These are needed at the minimum to implement the boundary conditions,
  // and are also used for computing the error norms
  [[nodiscard]] real flux_int_solution(const real x, const real y,
                                       const real time) const noexcept {
    const real vel_nabla_t_fi = flux_int_vel_nabla_T_sol(x, y, time);
    const real diffusion_fi   = flux_int_nabla2_T_sol(x, y, time);
    return vel_nabla_t_fi + diffusion_fi;
  }

  [[nodiscard]] real flux_int_vel_nabla_T_sol(const real x, const real y,
                                              const real time) const noexcept {
    const real u_dt_dy_fi =
        pi * u_0() * std::cos(2.0 * pi * x) * y * std::sin(pi * y);
    const real v_dt_dy_fi =
        pi * v_0() * x * std::cos(pi * x) * std::cos(2.0 * pi * y);
    return -T_0() * (u_dt_dy_fi + v_dt_dy_fi);
  }

  [[nodiscard]] real flux_int_nabla2_T_sol(const real x, const real y,
                                           const real time) const noexcept {
    return -T_0() * (2.0 * pi * pi / (reynolds * prandtl)) * std::cos(pi * x) *
           std::sin(pi * y);
  }

  // Use the exact solutions to implement the boundary conditions and also check
  // that the flux integral is correct
  [[nodiscard]] real solution(const real x, const real y, const real time)
      const noexcept {
    return T_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real source_sol(const real x, const real y, const real time)
      const noexcept {
    const real u_dx = u_0() * pi * std::cos(pi * x) * y;
    const real v_dy = v_0() * pi * std::sin(pi * y) * x;
    const real u_dy = u_0() * std::sin(pi * x);
    const real v_dx = v_0() * std::cos(pi * y);

    const real cross_term = u_dy + v_dx;
    return eckert / reynolds *
           (2.0 * (u_dx * u_dx + v_dy * v_dy) + cross_term * cross_term);
  }

  [[nodiscard]] real u(const real x, const real y) const noexcept {
    return u_0() * y * std::sin(pi * x);
  }

  [[nodiscard]] real v(const real x, const real y) const noexcept {
    return v_0() * x * std::cos(pi * y);
  }

  // To make filling the mesh with the initial solution easier
  [[nodiscard]] std::function<std::tuple<real, real, real>(real, real)>
  initial_solution_tuple() const noexcept {
    return [=](const real x, const real y) {
      return std::tuple<real, real, real>(solution(x, y, 0.0), u(x, y),
                                          v(x, y));
    };
  }

  [[nodiscard]] real solution_dx(const real x, const real y, const real time)
      const noexcept {
    return -pi * T_0() * std::sin(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy(const real x, const real y, const real time)
      const noexcept {
    return pi * T_0() * std::cos(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real solution_dx2(const real x, const real y, const real time)
      const noexcept {
    return -pi * pi * T_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy2(const real x, const real y, const real time)
      const noexcept {
    return solution_dx2(x, y, time);
  }

  [[nodiscard]] real boundary_x_0(const real y, const real time)
      const noexcept {
    return solution(x_min(), y, time);
  }

  [[nodiscard]] real boundary_x_1(const real y, const real time)
      const noexcept {
    return solution(x_max(), y, time);
  }

  [[nodiscard]] real boundary_y_0(const real x, const real time)
      const noexcept {
    return solution(x, y_min(), time);
  }

  [[nodiscard]] real boundary_y_1(const real x, const real time)
      const noexcept {
    return solution(x, y_max(), time);
  }

  [[nodiscard]] real boundary_dx_0(const real y, const real time)
      const noexcept {
    return solution_dx(x_min(), y, time);
  }

  [[nodiscard]] real boundary_dx_1(const real y, const real time)
      const noexcept {
    return solution_dx(x_min(), y, time);
  }

  [[nodiscard]] real boundary_dy_0(const real x, const real time)
      const noexcept {
    return solution_dy(x, y_min(), time);
  }

  [[nodiscard]] real boundary_dy_1(const real x, const real time)
      const noexcept {
    return solution_dy(x, y_max(), time);
  }

  constexpr SecondOrderCentered_Part1(const real T_0, const real u_0,
                                      const real v_0) noexcept
      : _t_0(T_0), _u_0(u_0), _v_0(v_0) {}

  constexpr SecondOrderCentered_Part1(
      const SecondOrderCentered_Part1 &src) noexcept
      : SecondOrderCentered_Part1(src.T_0(), src.u_0(), src.v_0()) {}

 protected:
  const real _t_0;
  const real _u_0, _v_0;
};

#endif  // _SPACE_DISC_HPP_
