
#ifndef _SPACE_DISC_HPP_
#define _SPACE_DISC_HPP_

#include "constants.hpp"

// Use the curiously repeated template parameter to swap out the order of the
// discretization in our assembly
template <typename _SpaceDisc>
class EnergyAssembly : public _SpaceDisc {
 public:
  // u = u_0 y \sin(\pi x)
  // v = v_0 x \cos(\pi y)
  using SpaceDisc = _SpaceDisc;

  constexpr EnergyAssembly(const real T_0, const real u_0,
                           const real v_0) noexcept
      : _u_0(u_0), _v_0(v_0), SpaceDisc(T_0) {}

  // [[nodiscard]] breaks clang-format here, and disabling clang-format here
  // doesn't seem to work
  constexpr real u_0() const noexcept { return _u_0; }

  [[nodiscard]] constexpr real v_0() const noexcept { return _v_0; }

  template <typename MeshT>
  void flux_assembly(const MeshT &initial, const MeshT &current, MeshT &next,
                     const real time, const real dt) const noexcept {
    for(int i = 0; i < initial.extent(0); i++) {
      for(int j = 0; j < initial.extent(1); j++) {
        const real x_deriv =
            (this->x_flux(current, i, j) - this->x_flux(current, i - 1, j)) /
            initial.dx();
        const real y_deriv =
            (this->y_flux(current, i, j) - this->y_flux(current, i, j - 1)) /
            initial.dy();
        const real x2_deriv =
            (this->dx_flux(current, i, j) - this->dx_flux(current, i - 1, j)) /
            initial.dx();
        const real y2_deriv =
            (this->dy_flux(current, i, j) - this->dy_flux(current, i, j - 1)) /
            initial.dy();
        next(i, j) =
            initial(i, j) - dt * (u_0 * x_deriv + v_0 * y_deriv -
                                  (x2_deriv + y2_deriv) / (reynolds * prandtl));
      }
    }
  }

 protected:
  const real _u_0, _v_0;
};

class SecondOrderCentered_Part1 {
  // T = T_0 \cos(\pi x) \sin(\pi y)

 public:
  constexpr SecondOrderCentered_Part1(const real T_0) noexcept : _t_0(T_0) {}

  constexpr SecondOrderCentered_Part1(
      const SecondOrderCentered_Part1 &src) noexcept
      : SecondOrderCentered_Part1(src._t_0){}

            [[nodiscard]] real solution(const real x, const real y) const
        noexcept {
    return _t_0 * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dx(const real x, const real y) const noexcept {
    return -pi * _t_0 * std::sin(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy(const real x, const real y) const noexcept {
    return pi * _t_0 * std::cos(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real boundary_x_0(const real y) const noexcept {
    return solution(0.0, y);
  }

  [[nodiscard]] real boundary_x_1(const real y) const noexcept {
    return solution(1.0, y);
  }

  [[nodiscard]] real boundary_y_0(const real x) const noexcept {
    return solution(x, 0.0);
  }

  [[nodiscard]] real boundary_y_1(const real x) const noexcept {
    return solution(x, 1.0);
  }

  [[nodiscard]] real boundary_dx_0(const real y) const noexcept {
    return solution_dx(0.0, y);
  }

  [[nodiscard]] real boundary_dx_1(const real y) const noexcept {
    return solution_dx(1.0, y);
  }

  [[nodiscard]] real boundary_dy_0(const real y) const noexcept {
    return solution_dx(0.0, y);
  }

  [[nodiscard]] real boundary_dy_1(const real y) const noexcept {
    return solution_dx(1.0, y);
  }

  // Using centered approximations to T_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] real x_flux(const MeshT &mesh, const int i, const int j) const
      noexcept {
    if(i == mesh.extent(0) - 1) {
      const real y = mesh.y_median(j);
      return boundary_x_1(y);
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      return boundary_x_0(y);
    } else {
      return (mesh(i, j) + mesh(i + 1, j)) / 2.0;
    }
  }

  // Using centered approximations to dT/dx_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] real dx_flux(const MeshT &mesh, const int i, const int j) const
      noexcept {
    if(i == mesh.extent(0) - 1) {
      const real y = mesh.y_median(j);
      return boundary_dx_1(y);
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      return boundary_dx_0(y);
    } else {
      return (mesh(i + 1, j) - mesh(i, j)) / (2.0 * mesh.dx());
    }
  }

  // Using centered approximations to T_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] real y_flux(const MeshT &mesh, const int i, const int j) const
      noexcept {
    if(j == mesh.extent(1) - 1) {
      const real x = mesh.x_median(i);
      return boundary_y_1(x);
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      return boundary_y_0(x);
    } else {
      return (mesh(i, j) + mesh(i, j + 1)) / (2.0 * mesh.dy());
    }
  }

  // Using centered approximations to dT/dy_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] real dy_flux(const MeshT &mesh, const int i, const int j) const
      noexcept {
    if(j == mesh.extent(1) - 1) {
      const real x = mesh.x_median(i);
      return boundary_dy_1(x);
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      return boundary_dy_0(x);
    } else {
      return (mesh(i, j + 1) - mesh(i, j)) / (2.0 * mesh.dy());
    }
  }

 protected:
  const real _t_0;
};

#endif  // _SPACE_DISC_HPP_
