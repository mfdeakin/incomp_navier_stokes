
#ifndef _SPACE_DISC_HPP_
#define _SPACE_DISC_HPP_

#include "boundaries.hpp"
#include "constants.hpp"

#include <memory>

// Use the curiously repeated template parameter to swap out the order of the
// discretization in our assembly
template <typename _SpaceDisc>
class[[nodiscard]] EnergyAssembly : public _SpaceDisc {
 public:
  // u = u_0 y \sin(\pi x)
  // v = v_0 x \cos(\pi y)
  using SpaceDisc = _SpaceDisc;

  static std::unique_ptr<BConds_Base> default_boundaries() noexcept {
    return std::make_unique<BConds_Part1>(1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  }

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
  constexpr EnergyAssembly(
      std::unique_ptr<BConds_Base> &&boundaries = default_boundaries(),
      const real diffusion                      = 1.0) noexcept
      : SpaceDisc(std::move(boundaries)), _diffuse_coeff(diffusion) {}

 protected:
  const real _diffuse_coeff;
};

class [[nodiscard]] SecondOrderCentered {
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
      return _boundaries->boundary_x_max(y, time) *
             _boundaries->u(_boundaries->x_max(), y);
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      return _boundaries->boundary_x_min(y, time) *
             _boundaries->u(_boundaries->x_min(), y);
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
      return _boundaries->boundary_y_max(x, time) *
             _boundaries->v(x, _boundaries->y_max());
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      return _boundaries->boundary_y_min(x, time) *
             _boundaries->v(x, _boundaries->y_min());
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
      const real y = mesh.y_median(j);
      const real T_right =
          2.0 * _boundaries->boundary_x_max(y, time) - mesh.Temp(i, j);
      return (T_right - mesh.Temp(i, j)) / mesh.dx();
    } else if(i == -1) {
      const real y = mesh.y_median(j);
      const real T_left =
          2.0 * _boundaries->boundary_x_min(y, time) - mesh.Temp(i + 1, j);
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
      const real x = mesh.x_median(i);
      const real T_above =
          2.0 * _boundaries->boundary_y_max(x, time) - mesh.Temp(i, j);
      return (T_above - mesh.Temp(i, j)) / mesh.dy();
    } else if(j == -1) {
      const real x = mesh.x_median(i);
      const real T_below =
          2.0 * _boundaries->boundary_y_min(x, time) - mesh.Temp(i, j + 1);
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
      return (mesh.u_vel(i + 1, j) - _boundaries->u(x_left, y)) /
             (2.0 * mesh.dx());
    } else if(i == mesh.x_dim() - 1) {
      const real x_right = mesh.x_median(i + 1);
      const real y       = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (_boundaries->u(x_right, y) - mesh.u_vel(i - 1, j)) /
             (2.0 * mesh.dx());
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
      return (mesh.u_vel(i, j + 1) - _boundaries->u(x, y_below)) /
             (2.0 * mesh.dy());
    } else if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real y_above = mesh.y_median(j + 1);
      // Use our exact solution to u outside of the boundaries
      return (_boundaries->u(x, y_above) - mesh.u_vel(i, j - 1)) /
             (2.0 * mesh.dy());
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
      return (mesh.v_vel(i + 1, j) - _boundaries->v(x_left, y)) /
             (2.0 * mesh.dx());
    } else if(i == mesh.x_dim() - 1) {
      const real x_right = mesh.x_median(i + 1);
      const real y       = mesh.y_median(j);
      // Use our exact solution to u outside of the boundaries
      return (_boundaries->v(x_right, y) - mesh.v_vel(i - 1, j)) /
             (2.0 * mesh.dx());
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
      return (mesh.v_vel(i, j + 1) - _boundaries->v(x, y_below)) /
             (2.0 * mesh.dy());
    } else if(j == mesh.y_dim() - 1) {
      const real x       = mesh.x_median(i);
      const real y_above = mesh.y_median(j + 1);
      // Use our exact solution to u outside of the boundaries
      return (_boundaries->v(x, y_above) - mesh.v_vel(i, j - 1)) /
             (2.0 * mesh.dy());
    } else {
      return (mesh.v_vel(i, j + 1) - mesh.v_vel(i, j - 1)) / (2.0 * mesh.dy());
    }
  }

  const BConds_Base *boundaries() const noexcept { return _boundaries.get(); }

  SecondOrderCentered(std::unique_ptr<BConds_Base> && boundaries) noexcept
      : _boundaries(std::move(boundaries)) {}

  SecondOrderCentered() = delete;

 protected:
  std::unique_ptr<BConds_Base> _boundaries;
};

#endif  // _SPACE_DISC_HPP_
