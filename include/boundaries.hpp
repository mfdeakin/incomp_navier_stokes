
#ifndef _BOUNDARIES_HPP_
#define _BOUNDARIES_HPP_

#include "constants.hpp"
#include "multivar.hpp"

#include <cmath>
#include <functional>

class BConds_Base {
 public:
  [[nodiscard]] constexpr real x_min() const noexcept { return _x_min; }
  [[nodiscard]] constexpr real y_min() const noexcept { return _y_min; }
  [[nodiscard]] constexpr real x_max() const noexcept { return _x_max; }
  [[nodiscard]] constexpr real y_max() const noexcept { return _y_max; }

  [[nodiscard]] constexpr real P_0() const noexcept { return _p_0; }
  [[nodiscard]] constexpr real u_0() const noexcept { return _u_0; }
  [[nodiscard]] constexpr real v_0() const noexcept { return _v_0; }
  [[nodiscard]] constexpr real beta() const noexcept { return _beta; }

  constexpr BConds_Base(const real P_0, const real u_0, const real v_0,
                        const real beta, const real x_min = 0.0,
                        const real x_max = 1.0, const real y_min = 0.0,
                        const real y_max = 1.0)
      : _p_0(P_0),
        _u_0(u_0),
        _v_0(v_0),
        _beta(beta),
        _x_min(x_min),
        _x_max(x_max),
        _y_min(y_min),
        _y_max(y_max) {}

  constexpr BConds_Base(const BConds_Base &src)
      : BConds_Base(src.P_0(), src.u_0(), src.v_0(), src.beta(), src.x_min(),
                    src.x_max(), src.y_min(), src.y_max()) {}

  virtual ~BConds_Base() {}

  // Functions needed by every implementation of the boundary conditions
  // These will probably be inlined, and if not, they're only needed for
  // boundaries calculations, so the performance cost of an extra dereference
  // and function call shouldn't be that bad
  template <typename MeshT>
  void init_mesh(MeshT &mesh) const noexcept {
    for(int i = 0; i < mesh.x_dim(); i++) {
      for(int j = 0; j < mesh.y_dim(); j++) {
        const real x     = mesh.x_median(i);
        const real y     = mesh.y_median(j);
        mesh.press(i, j) = pressure_initial(x, y, mesh.dx(), mesh.dy());
        mesh.u_vel(i, j) = u_vel_initial(x, y, mesh.dx(), mesh.dy());
        mesh.v_vel(i, j) = v_vel_initial(x, y, mesh.dx(), mesh.dy());
      }
    }
  }

  triple initial_conds(const real x, const real y, const real dx,
                       const real dy) const noexcept {
    return {pressure_initial(x, y, dx, dy), u_vel_initial(x, y, dx, dy),
            v_vel_initial(x, y, dx, dy)};
  };

  triple boundary_x_min(const real y, const real time) const noexcept {
    return {pressure_boundary_x_min(y, time), u_vel_boundary_x_min(y, time),
            v_vel_boundary_x_min(y, time)};
  }
  triple boundary_x_max(const real y, const real time) const noexcept {
    return {pressure_boundary_x_max(y, time), u_vel_boundary_x_max(y, time),
            v_vel_boundary_x_max(y, time)};
  }
  triple boundary_y_min(const real x, const real time) const noexcept {
    return {pressure_boundary_y_min(x, time), u_vel_boundary_y_min(x, time),
            v_vel_boundary_y_min(x, time)};
  }
  triple boundary_y_max(const real x, const real time) const noexcept {
    return {pressure_boundary_y_max(x, time), u_vel_boundary_y_max(x, time),
            v_vel_boundary_y_max(x, time)};
  }

  virtual real pressure_initial(const real x, const real y, const real dx,
                                const real dy) const noexcept = 0;
  virtual real pressure_boundary_x_min(const real y, const real time) const
      noexcept = 0;
  virtual real pressure_boundary_x_max(const real y, const real time) const
      noexcept = 0;
  virtual real pressure_boundary_y_min(const real x, const real time) const
      noexcept = 0;
  virtual real pressure_boundary_y_max(const real x, const real time) const
      noexcept = 0;

  virtual real u_vel_initial(const real x, const real y, const real dx,
                             const real dy) const noexcept = 0;
  virtual real u_vel_boundary_x_min(const real y, const real time) const
      noexcept = 0;
  virtual real u_vel_boundary_x_max(const real y, const real time) const
      noexcept = 0;
  virtual real u_vel_boundary_y_min(const real x, const real time) const
      noexcept = 0;
  virtual real u_vel_boundary_y_max(const real x, const real time) const
      noexcept = 0;

  virtual real v_vel_initial(const real x, const real y, const real dx,
                             const real dy) const noexcept = 0;
  virtual real v_vel_boundary_x_min(const real y, const real time) const
      noexcept = 0;
  virtual real v_vel_boundary_x_max(const real y, const real time) const
      noexcept = 0;
  virtual real v_vel_boundary_y_min(const real x, const real time) const
      noexcept = 0;
  virtual real v_vel_boundary_y_max(const real x, const real time) const
      noexcept = 0;

 protected:
  real _p_0;
  real _u_0, _v_0;
  real _beta;
  real _x_min, _x_max, _y_min, _y_max;
};

// dP/dt + (du/dx + dv/dy) / beta = 0.0
// du/dt + d(u^2)/dx + d(uv)/dy = (d^2u/dx^2 + d^2u/dy^2) / Re - dP/dx
// dv/dt + d(uv)/dx + d(v^2)/dy = (d^2v/dx^2 + d^2v/dy^2) / Re - dP/dy

class BConds_Part1 : public BConds_Base {
 public:
  // P = P_0 \cos(\pi x) \sin(\pi y)
  // u = u_0 y \sin(\pi x)
  // v = v_0 x \cos(\pi y)

  // The rest of the methods are just related to the analytic solution
  // These are needed at the minimum to implement the boundary conditions,
  // and are also used for computing the error norms
  // Use the exact solutions to implement the boundary conditions and also check
  // that the flux integral is correct
  [[nodiscard]] real pressure(const real x, const real y, const real time) const
      noexcept {
    return P_0() * std::cos(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real pressure_avg_initial(const real x, const real y,
                                          const real dx, const real dy) const
      noexcept {
    return P_0() / (pi * pi * dx * dy) *
           (std::sin(pi * (x + dx / 2.0)) - std::sin(pi * (x - dx / 2.0))) *
           (std::sin(pi * (y + dy / 2.0)) - std::sin(pi * (y - dy / 2.0)));
  }

  [[nodiscard]] real u(const real x, const real y, const real time) const
      noexcept {
    return u_0() * std::sin(pi * x) * std::sin(2.0 * pi * y);
  }

  [[nodiscard]] real v(const real x, const real y, const real time) const
      noexcept {
    return v_0() * std::sin(2.0 * pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real pressure_dx(const real x, const real y,
                                 const real time) const noexcept {
    return -pi * P_0() * std::sin(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real pressure_dy(const real x, const real y,
                                 const real time) const noexcept {
    return -pi * P_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real u_dx(const real x, const real y, const real time) const
      noexcept {
    return u_0() * pi * std::cos(pi * x) * std::sin(2.0 * pi * y);
  }

  [[nodiscard]] real v_dy(const real x, const real y, const real time) const
      noexcept {
    return v_0() * pi * std::sin(2.0 * pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real u2_dx(const real x, const real y, const real time) const
      noexcept {
    return -u_0() * u_0() * pi * std::sin(2.0 * pi * x) *
           std::sin(2.0 * pi * y) * std::sin(2.0 * pi * y);
  }

  [[nodiscard]] real v2_dx(const real x, const real y, const real time) const
      noexcept {
    return -v_0() * v_0() * pi * std::sin(2.0 * pi * y) *
           std::sin(2.0 * pi * x) * std::sin(2.0 * pi * x);
  }

  [[nodiscard]] real uv_dx(const real x, const real y, const real time) const
      noexcept {
    return -u_0() * v_0() * std::sin(pi * x) * std::sin(2.0 * pi * x) *
           (std::cos(pi * y) * std::sin(2.0 * pi * y) +
            2.0 * std::cos(2.0 * pi * y) * std::sin(pi * y));
  }

  [[nodiscard]] real uv_dy(const real x, const real y, const real time) const
      noexcept {
    return -u_0() * v_0() * std::sin(pi * y) * std::sin(2.0 * pi * y) *
           (std::cos(pi * x) * std::sin(2.0 * pi * x) +
            2.0 * std::cos(2.0 * pi * x) * std::sin(pi * x));
  }

  [[nodiscard]] real u_dx2(const real x, const real y, const real time) const
      noexcept {
    return -u_0() * pi * pi * std::sin(pi * x) * std::sin(2.0 * pi * y);
  }

  [[nodiscard]] real v_dx2(const real x, const real y, const real time) const
      noexcept {
    return -v_0() * 4.0 * pi * pi * std::sin(2.0 * pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real u_dy2(const real x, const real y, const real time) const
      noexcept {
    return -u_0() * 4.0 * pi * pi * std::sin(pi * x) * std::sin(2.0 * pi * y);
  }

  [[nodiscard]] real v_dy2(const real x, const real y, const real time) const
      noexcept {
    return -v_0() * pi * pi * std::sin(pi * y) * std::sin(2.0 * pi * x);
  }

  [[nodiscard]] triple solution(const real x, const real y,
                                const real time) const noexcept {
    return {pressure(x, y, time), u(x, y, time), v(x, y, time)};
  }

  [[nodiscard]] triple flux_int_solution(const real x, const real y,
                                         const real time) const noexcept {
    const real cx = std::cos(pi * x), sx = std::sin(pi * x),
               c2x = std::cos(2.0 * pi * x), s2x = std::sin(2.0 * pi * x);
    const real cy = std::cos(pi * y), sy = std::sin(pi * y),
               c2y = std::cos(2.0 * pi * y), s2y = std::sin(2.0 * pi * y);
    const real p_term = -pi / beta() * (u_0() * cx * s2y + v_0() * s2x * cy);
    const real u_term =
        pi * (P_0() * sx * cy -
              u_0() * (u_0() * s2x * s2y * s2y +
                       v_0() * sx * s2x * (cy * s2y + 2.0 * c2y * sy) +
                       5.0 * pi * sx * s2y / reynolds));
    const real v_term =
        pi * (P_0() * sy * cx -
              v_0() * (v_0() * s2y * s2x * s2x +
                       u_0() * sy * s2y * (cx * s2x + 2.0 * c2x * sx) +
                       5.0 * pi * sy * s2x / reynolds));
    return {p_term, u_term, v_term};
  }

  template <typename MeshT>
  void flux_int_fill(MeshT &mesh) {
    for(int i = 0; i < mesh.x_dim(); i++) {
      for(int j = 0; j < mesh.y_dim(); j++) {
        const real x = mesh.x_median(i);
        const real y = mesh.y_median(j);

        const auto [p, u, v] = flux_int_solution(x, y, 0.0);
        mesh.press(i, j)     = p;
        mesh.u_vel(i, j)     = u;
        mesh.v_vel(i, j)     = v;
      }
    }
  }

  [[nodiscard]] real pressure_initial(const real x, const real y, const real dx,
                                      const real dy) const noexcept {
    return pressure_avg_initial(x, y, dx, dy);
  }

  [[nodiscard]] real pressure_boundary_x_min(const real y,
                                             const real time) const noexcept {
    return pressure(x_min(), y, time);
  }

  [[nodiscard]] real pressure_boundary_x_max(const real y,
                                             const real time) const noexcept {
    return pressure(x_max(), y, time);
  }

  [[nodiscard]] real pressure_boundary_y_min(const real x,
                                             const real time) const noexcept {
    return pressure(x, y_min(), time);
  }

  [[nodiscard]] real pressure_boundary_y_max(const real x,
                                             const real time) const noexcept {
    return pressure(x, y_max(), time);
  }

  [[nodiscard]] real u_vel_initial(const real x, const real y, const real dx,
                                   const real dy) const noexcept {
    return u(x, y, 0.0);
  }

  [[nodiscard]] real u_vel_boundary_x_min(const real y, const real time) const
      noexcept {
    return u(x_min(), y, time);
  }

  [[nodiscard]] real u_vel_boundary_x_max(const real y, const real time) const
      noexcept {
    return u(x_max(), y, time);
  }

  [[nodiscard]] real u_vel_boundary_y_min(const real x, const real time) const
      noexcept {
    return u(x, y_min(), time);
  }

  [[nodiscard]] real u_vel_boundary_y_max(const real x, const real time) const
      noexcept {
    return u(x, y_max(), time);
  }

  [[nodiscard]] real v_vel_initial(const real x, const real y, const real dx,
                                   const real dy) const noexcept {
    return v(x, y, 0.0);
  }

  [[nodiscard]] real v_vel_boundary_x_min(const real y, const real time) const
      noexcept {
    return v(x_min(), y, time);
  }

  [[nodiscard]] real v_vel_boundary_x_max(const real y, const real time) const
      noexcept {
    return v(x_max(), y, time);
  }

  [[nodiscard]] real v_vel_boundary_y_min(const real x, const real time) const
      noexcept {
    return v(x, y_min(), time);
  }

  [[nodiscard]] real v_vel_boundary_y_max(const real x, const real time) const
      noexcept {
    return v(x, y_max(), time);
  }

  constexpr BConds_Part1(const real P_0, const real u_0, const real v_0,
                         const real beta, const real x_min = 0.0,
                         const real x_max = 1.0, const real y_min = 0.0,
                         const real y_max = 1.0)
      : BConds_Base(P_0, u_0, v_0, beta, x_min, x_max, y_min, y_max) {}

  constexpr BConds_Part1(const BConds_Base &src) : BConds_Base(src) {}
};

#endif  // _BOUNDARIES_HPP_
