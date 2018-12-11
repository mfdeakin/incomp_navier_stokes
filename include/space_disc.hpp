
#ifndef _SPACE_DISC_HPP_
#define _SPACE_DISC_HPP_

#include "boundaries.hpp"
#include "constants.hpp"

#include <memory>
#include <type_traits>

template <typename _SpaceDisc>
class [[nodiscard]] INSAssembly : public _SpaceDisc {
 public:
  using SpaceDisc = _SpaceDisc;
  using BConds    = typename SpaceDisc::BConds;

  template <typename MeshT>
  [[nodiscard]] triple flux_integral(const MeshT &mesh, int i, int j,
                                     const real time) const noexcept {
    const real u_right = this->u_x_flux(mesh, i, j, time);
    const real u_left  = this->u_x_flux(mesh, i - 1, j, time);
    const real v_above = this->v_y_flux(mesh, i, j, time);
    const real v_below = this->v_y_flux(mesh, i, j - 1, time);

    const real fp_u_x_deriv = (u_right - u_left) / mesh.dx();
    const real gp_v_y_deriv = (v_above - v_below) / mesh.dy();
    const real p_term =
        (fp_u_x_deriv + gp_v_y_deriv) / this->boundaries().beta();

    const real reynolds = this->boundaries().reynolds();

    const auto diag_term = [=](const real vel2, const real p,
                              const real vel_deriv) {
      return vel2 + p - vel_deriv / reynolds;
    };

    const auto cross_term = [=](const real v1_v2, const real dv1_dx2) {
      return v1_v2 - dv1_dx2 / reynolds;
    };

    const real u2_right = u_right * u_right;
    const real u2_left  = u_left * u_left;
    const real p_right  = this->press_x_flux(mesh, i, j, time);
    const real p_left   = this->press_x_flux(mesh, i - 1, j, time);
    const real du_right = this->du_x_flux(mesh, i, j, time);
    const real du_left  = this->du_x_flux(mesh, i - 1, j, time);
    const real u_f_term = (diag_term(u2_right, p_right, du_right) -
                           diag_term(u2_left, p_left, du_left)) /
                          mesh.dx();

    const real u_above  = this->u_y_flux(mesh, i, j, time);
    const real u_below  = this->u_y_flux(mesh, i, j - 1, time);
    const real uv_above = u_above * v_above;
    const real uv_below = u_below * v_below;
    const real du_above = this->du_y_flux(mesh, i, j, time);
    const real du_below = this->du_y_flux(mesh, i, j - 1, time);
    const real u_g_term =
        (cross_term(uv_above, du_above) - cross_term(uv_below, du_below)) /
        mesh.dy();

    const real v2_above = v_above * v_above;
    const real v2_below = v_below * v_below;
    const real p_above  = this->press_y_flux(mesh, i, j, time);
    const real p_below  = this->press_y_flux(mesh, i, j - 1, time);
    const real dv_above = this->dv_y_flux(mesh, i, j, time);
    const real dv_below = this->dv_y_flux(mesh, i, j - 1, time);
    const real v_f_term = (diag_term(v2_above, p_above, dv_above) -
                           diag_term(v2_below, p_below, dv_below)) /
                          mesh.dy();

    const real v_right  = this->v_x_flux(mesh, i, j, time);
    const real v_left   = this->v_x_flux(mesh, i - 1, j, time);
    const real uv_right = u_right * v_right;
    const real uv_left  = u_left * v_left;
    const real dv_right = this->dv_x_flux(mesh, i, j, time);
    const real dv_left  = this->dv_x_flux(mesh, i - 1, j, time);
    const real v_g_term =
        (cross_term(uv_right, dv_right) - cross_term(uv_left, dv_left)) /
        mesh.dx();

    return {-p_term, -(u_f_term + u_g_term), -(v_f_term + v_g_term)};
  }

  template <typename MeshT>
  void flux_assembly(const MeshT &initial, const MeshT &current, MeshT &next,
                     const real time, const real dt) const noexcept {
    for(int i = 0; i < initial.x_dim(); i++) {
      for(int j = 0; j < initial.y_dim(); j++) {
        const auto &[p_flux, u_flux, v_flux] =
            flux_integral(current, i, j, time);
        next.press(i, j) = initial.press(i, j) + dt * p_flux;
        next.u_vel(i, j) = initial.u_vel(i, j) + dt * u_flux;
        next.v_vel(i, j) = initial.v_vel(i, j) + dt * v_flux;
      }
    }
  }

  INSAssembly(const BConds &boundaries) : SpaceDisc(boundaries) {}
};

template <typename _BConds>
class [[nodiscard]] SecondOrderCentered {
 public:
  using BConds = _BConds;

  // Terms for implicit euler
  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian jacobian_x_base(const MeshT &mesh, int i,
                                                   int j) const noexcept {
    Jacobian j_x((Jacobian::ZeroTag()));
    j_x(0, 0) = 0.0;
    j_x(0, 1) = 0.5 / boundaries().beta();
    j_x(0, 2) = 0.0;

    j_x(1, 0) = 0.5;
    j_x(1, 1) = 0.5 * (mesh.u_vel(i + 1, j) + mesh.u_vel(i, j));
    j_x(1, 2) = 0.0;

    j_x(2, 0) = 0.0;
    j_x(2, 1) = 0.25 * (mesh.v_vel(i + 1, j) + mesh.v_vel(i, j));
    j_x(2, 2) = 0.25 * (mesh.u_vel(i + 1, j) + mesh.u_vel(i, j));
    return j_x;
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian jacobian_x_0(const MeshT &mesh, int i, int j)
      const noexcept {
    // This is the Jacobian of F_{i+1/2,j} wrt U_{i,j}
    Jacobian b_x(jacobian_x_base(mesh, i, j));
    const real deriv_term = 1.0 / (boundaries().reynolds() * mesh.dx());
    b_x(1, 1) += deriv_term;
    b_x(2, 2) += deriv_term;
    return b_x;
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian jacobian_x_p1(const MeshT &mesh, int i,
                                                 int j) const noexcept {
    // This is the Jacobian of F_{i+1/2,j} wrt U_{i+1,j}
    Jacobian b_x(jacobian_x_base(mesh, i, j));
    const real deriv_term = 1.0 / (boundaries().reynolds() * mesh.dx());
    b_x(1, 1) -= deriv_term;
    b_x(2, 2) -= deriv_term;
    return b_x;
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian Dx_p1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(i < mesh.x_dim() - 1) {
      return jacobian_x_p1(mesh, i, j) * (1.0 / mesh.dx());
    } else {
      return Jacobian(Jacobian::ZeroTag());
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian Dx_0(const MeshT &mesh, int i, int j)
      const noexcept {
    return (jacobian_x_0(mesh, i, j) - jacobian_x_p1(mesh, i - 1, j)) *
           (1.0 / mesh.dx());
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian Dx_m1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(i > 0) {
      return jacobian_x_0(mesh, i - 1, j) * (-1.0 / mesh.dx());
    } else {
      return Jacobian(Jacobian::ZeroTag());
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian jacobian_y_base(const MeshT &mesh, int i,
                                                   int j) const noexcept {
    Jacobian j_y((Jacobian::ZeroTag()));
    j_y(0, 0) = 0.0;
    j_y(0, 1) = 0.0;
    j_y(0, 2) = 0.5 / boundaries().beta();

    j_y(1, 0) = 0.0;
    j_y(1, 1) = 0.25 * (mesh.v_vel(i, j + 1) + mesh.v_vel(i, j)) +
                1.0 / (boundaries().reynolds() * mesh.dy());
    j_y(1, 2) = 0.25 * (mesh.u_vel(i + 1, j) + mesh.u_vel(i, j));

    j_y(2, 0) = 0.5;
    j_y(2, 1) = 0.0;
    j_y(2, 2) = 0.5 * (mesh.v_vel(i, j + 1) + mesh.v_vel(i, j)) +
                1.0 / (boundaries().reynolds() * mesh.dy());
    return j_y;
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian jacobian_y_0(const MeshT &mesh, int i, int j)
      const noexcept {
    // This is the Jacobian of F_{i+1/2,j} wrt U_{i,j}
    Jacobian j_y(jacobian_x_base(mesh, i, j));
    const real deriv_term = 1.0 / (boundaries().reynolds() * mesh.dy());
    j_y(1, 1) += deriv_term;
    j_y(2, 2) += deriv_term;
    return j_y;
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian jacobian_y_p1(const MeshT &mesh, int i,
                                                 int j) const noexcept {
    // This is the Jacobian of F_{i+1/2,j} wrt U_{i+1,j}
    Jacobian j_y(jacobian_x_base(mesh, i, j));
    const real deriv_term = 1.0 / (boundaries().reynolds() * mesh.dy());
    j_y(1, 1) -= deriv_term;
    j_y(2, 2) -= deriv_term;
    return j_y;
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian Dy_p1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(j < mesh.y_dim() - 1) {
      return jacobian_y_p1(mesh, i, j) * (1.0 / mesh.dy());
    } else {
      return Jacobian(Jacobian::ZeroTag());
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian Dy_0(const MeshT &mesh, int i, int j)
      const noexcept {
    return (jacobian_y_0(mesh, i, j) - jacobian_y_p1(mesh, i, j - 1)) *
           (1.0 / mesh.dy());
  }

  template <typename MeshT>
  [[nodiscard]] constexpr Jacobian Dy_m1(const MeshT &mesh, int i, int j)
      const noexcept {
    if(j > 0) {
      return jacobian_y_0(mesh, i, j - 1) * (-1.0 / mesh.dy());
    } else {
      return Jacobian(Jacobian::ZeroTag());
    }
  }

  template <typename MeshT, typename Get>
  [[nodiscard]] constexpr real get_value_x_flux(
      const MeshT &mesh, const Get &get_value, int i, int j) const noexcept {
    return (get_value(mesh, i, j) + get_value(mesh, i + 1, j)) / 2.0;
  }

  template <typename MeshT, typename Get>
  [[nodiscard]] constexpr real get_value_y_flux(
      const MeshT &mesh, const Get &get_value, int i, int j) const noexcept {
    return (get_value(mesh, i, j) + get_value(mesh, i, j + 1)) / 2.0;
  }

  template <typename MeshT, typename Get>
  [[nodiscard]] constexpr real get_deriv_x_flux(
      const MeshT &mesh, const Get &get_value, int i, int j) const noexcept {
    return (get_value(mesh, i + 1, j) - get_value(mesh, i, j)) / mesh.dx();
  }

  template <typename MeshT, typename Get>
  [[nodiscard]] constexpr real get_deriv_y_flux(
      const MeshT &mesh, const Get &get_value, int i, int j) const noexcept {
    return (get_value(mesh, i, j + 1) - get_value(mesh, i, j)) / mesh.dy();
  }

  // Centered FV approximation to P_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real press_x_flux(const MeshT &mesh, const int i,
                                            const int j, const real time)
      const noexcept {
    return get_value_x_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().pressure_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to P_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real press_y_flux(const MeshT &mesh, const int i,
                                            const int j, const real time)
      const noexcept {
    return get_value_y_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().pressure_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to u_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real u_x_flux(const MeshT &mesh, const int i,
                                        const int j, const real time)
      const noexcept {
    return get_value_x_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().u_vel_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to u_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real u_y_flux(const MeshT &mesh, const int i,
                                        const int j, const real time)
      const noexcept {
    return get_value_y_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().u_vel_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to v_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real v_x_flux(const MeshT &mesh, const int i,
                                        const int j, const real time)
      const noexcept {
    return get_value_x_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().v_vel_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to v_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real v_y_flux(const MeshT &mesh, const int i,
                                        const int j, const real time)
      const noexcept {
    return get_value_y_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().v_vel_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to du_dx_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real du_x_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    return get_deriv_x_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().u_vel_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to du_dy_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real du_y_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    return get_deriv_y_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().u_vel_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to dv_dx_{i+1/2, j}
  template <typename MeshT>
  [[nodiscard]] constexpr real dv_x_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    return get_deriv_x_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().v_vel_at(mesh, time, i, j);
                            },
                            i, j);
  }

  // Centered FV approximation to dv_dy_{i, j+1/2}
  template <typename MeshT>
  [[nodiscard]] constexpr real dv_y_flux(const MeshT &mesh, const int i,
                                         const int j, const real time)
      const noexcept {
    return get_deriv_y_flux(mesh,
                            [=](const MeshT &mesh, const int i, const int j) {
                              return boundaries().v_vel_at(mesh, time, i, j);
                            },
                            i, j);
  }

  SecondOrderCentered() = delete;
  SecondOrderCentered(const BConds &bounds) : _boundaries(bounds) {}

  const BConds &boundaries() const noexcept { return _boundaries; }

 protected:
  BConds _boundaries;
};

#endif  // _SPACE_DISC_HPP_
