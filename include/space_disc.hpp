
#ifndef _SPACE_DISC_HPP_
#define _SPACE_DISC_HPP_

#include "constants.hpp"

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
  [[nodiscard]] real flux_integral(const MeshT &mesh, int i, int j)
      const noexcept {
    const real x_deriv =
        (this->uT_x_flux(mesh, i, j) - this->uT_x_flux(mesh, i - 1, j)) /
        mesh.dx();
    const real y_deriv =
        (this->vT_y_flux(mesh, i, j) - this->vT_y_flux(mesh, i, j - 1)) /
        mesh.dy();

    const real x2_deriv =
        (this->dx_flux(mesh, i, j) - this->dx_flux(mesh, i - 1, j)) / mesh.dx();
    const real y2_deriv =
        (this->dy_flux(mesh, i, j) - this->dy_flux(mesh, i, j - 1)) / mesh.dy();

    return -x_deriv - y_deriv +
           _diffuse_coeff * (x2_deriv + y2_deriv) / (reynolds * prandtl);
  }

  template <typename MeshT>
  void flux_assembly(const MeshT &initial, const MeshT &current, MeshT &next,
                     const real time, const real dt) const noexcept {
    for(int i = 0; i < initial.extent(0); i++) {
      for(int j = 0; j < initial.extent(1); j++) {
        next(i, j) = initial(i, j) - dt * flux_integral(current, i, j);
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

class [[nodiscard]] SecondOrderCentered_Part1 {
 public:
  // Using centered approximations to (u T)_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real uT_x_flux(const MeshT &mesh, const int i,
                                         const int j) const noexcept {
    if(i == mesh.x_dim() - 1) {
      const real y = mesh.y_median(j);
      return boundary_x_1(y);
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      return boundary_x_0(y);
    } else {
      return (mesh.Temp(i, j) * mesh.u_vel(i, j) +
              mesh.Temp(i + 1, j) * mesh.u_vel(i + 1, j)) /
             2.0;
    }
  }

  // Using centered approximations to dT/dx_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dx_flux(const MeshT &mesh, const int i,
                                       const int j) const noexcept {
    if(i == mesh.x_dim() - 1) {
      const real y = mesh.y_median(j);
      return boundary_dx_1(y);
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      return boundary_dx_0(y);
    } else {
      return (mesh.Temp(i + 1, j) - mesh.Temp(i, j)) / mesh.dx();
    }
  }

  // Using centered approximations to T_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real vT_y_flux(const MeshT &mesh, const int i,
                                         const int j) const noexcept {
    if(j == mesh.y_dim() - 1) {
      const real x = mesh.x_median(i);
      return boundary_y_1(x);
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      return boundary_y_0(x);
    } else {
      return (mesh.Temp(i, j) * mesh.v_vel(i, j) +
              mesh.Temp(i, j + 1) * mesh.v_vel(i, j + 1)) /
             2.0;
    }
  }

  // Using centered approximations to dT/dy_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real dy_flux(const MeshT &mesh, const int i,
                                       const int j) const noexcept {
    if(j == mesh.y_dim() - 1) {
      const real x = mesh.x_median(i);
      return boundary_dy_1(x);
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      return boundary_dy_0(x);
    } else {
      return (mesh.Temp(i, j + 1) - mesh.Temp(i, j)) / mesh.dy();
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

  [[nodiscard]] real flux_int_solution(const real x, const real y)
      const noexcept {
    const real u_dt_dx_fi =
        pi * u_0() * std::cos(2.0 * pi * x) * y * std::sin(pi * y);
    const real v_dt_dy_fi =
        pi * v_0() * x * std::cos(pi * x) * std::cos(2.0 * pi * y);
    const real diffusion_fi = (2.0 * pi * pi / (reynolds * prandtl)) *
                              std::cos(pi * x) * std::sin(pi * y);
    return T_0() * (u_dt_dx_fi + v_dt_dy_fi + diffusion_fi);
  }

  // Use the exact solutions to implement the boundary conditions and also check
  // that the flux integral is correct
  [[nodiscard]] real solution(const real x, const real y) const noexcept {
    return T_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real u(const real x, const real y) const noexcept {
    return u_0() * y * std::sin(pi * x);
  }

  [[nodiscard]] real v(const real x, const real y) const noexcept {
    return v_0() * x * std::cos(pi * y);
  }

  [[nodiscard]] real solution_dx(const real x, const real y) const noexcept {
    return -pi * T_0() * std::sin(pi * x) * std::sin(pi * y);
  }

  // To make filling the mesh with the initial solution easier
  [[nodiscard]] std::function<std::tuple<real, real, real>(real, real)>
  solution_tuple() const noexcept {
    return [=](const real x, const real y) {
      return std::tuple<real, real, real>(solution(x, y), u(x, y), v(x, y));
    };
  }

  [[nodiscard]] real solution_dy(const real x, const real y) const noexcept {
    return pi * T_0() * std::cos(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real solution_dx2(const real x, const real y) const noexcept {
    return -pi * pi * T_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy2(const real x, const real y) const noexcept {
    return solution_dx2(x, y);
  }

  [[nodiscard]] real boundary_x_0(const real y) const noexcept {
    return solution(x_min(), y) * u(x_min(), y);
  }

  [[nodiscard]] real boundary_x_1(const real y) const noexcept {
    return solution(x_max(), y) * u(x_max(), y);
  }

  [[nodiscard]] real boundary_y_0(const real x) const noexcept {
    return solution(x, y_min()) * v(x, y_min());
  }

  [[nodiscard]] real boundary_y_1(const real x) const noexcept {
    return solution(x, y_max()) * v(x, y_max());
  }

  [[nodiscard]] real boundary_dx_0(const real y) const noexcept {
    return solution_dx(x_min(), y);
  }

  [[nodiscard]] real boundary_dx_1(const real y) const noexcept {
    return solution_dx(x_min(), y);
  }

  [[nodiscard]] real boundary_dy_0(const real x) const noexcept {
    return solution_dy(x, y_min());
  }

  [[nodiscard]] real boundary_dy_1(const real x) const noexcept {
    return solution_dy(x, y_max());
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
