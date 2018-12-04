
#ifndef _BOUNDARIES_HPP_
#define _BOUNDARIES_HPP_

#include "constants.hpp"

#include <cmath>
#include <functional>

class BConds_Base {
 public:
  [[nodiscard]] constexpr real x_min() const noexcept { return _x_min; }
  [[nodiscard]] constexpr real y_min() const noexcept { return _y_min; }
  [[nodiscard]] constexpr real x_max() const noexcept { return _x_max; }
  [[nodiscard]] constexpr real y_max() const noexcept { return _y_max; }

  [[nodiscard]] constexpr real T_0() const noexcept { return _t_0; }

  [[nodiscard]] constexpr real u_0() const noexcept { return _u_0; }

  [[nodiscard]] constexpr real v_0() const noexcept { return _v_0; }

  // To make filling the mesh with the initial solution easier
  [[nodiscard]] std::function<std::tuple<real, real, real>(real, real)>
  initial_solution_tuple() const noexcept {
    return [=](const real x, const real y) {
      return std::tuple<real, real, real>(initial_conds(x, y), u(x, y),
                                          v(x, y));
    };
  }

  constexpr BConds_Base(const real T_0, const real u_0, const real v_0,
                        const real x_min = 0.0, const real x_max = 1.0,
                        const real y_min = 0.0, const real y_max = 1.0)
      : _t_0(T_0),
        _u_0(u_0),
        _v_0(v_0),
        _x_min(x_min),
        _x_max(x_max),
        _y_min(y_min),
        _y_max(y_max) {}

  constexpr BConds_Base(const BConds_Base &src)
      : BConds_Base(src.T_0(), src.u_0(), src.v_0(), src.x_min(), src.x_max(),
                    src.y_min(), src.y_max()) {}

  virtual ~BConds_Base() {}

  // Functions needed by every implementation of the boundary conditions
  // These will probably be inlined, and if not, they're only needed for
  // boundaries calculations and initialization, so who cares about the
  // performance cost of an extra dereference and function call?
  virtual real initial_conds(const real x, const real y) const noexcept = 0;

  virtual real boundary_x_min(const real y, const real time) const noexcept = 0;
  virtual real boundary_x_max(const real y, const real time) const noexcept = 0;
  virtual real boundary_y_min(const real x, const real time) const noexcept = 0;
  virtual real boundary_y_max(const real x, const real time) const noexcept = 0;

  virtual real u(const real x, const real y) const noexcept = 0;
  virtual real v(const real x, const real y) const noexcept = 0;

 protected:
  real _t_0;
  real _u_0, _v_0;
  real _x_min, _x_max, _y_min, _y_max;
};

class BConds_Part1 : public BConds_Base {
 public:
  // T = T_0 \cos(\pi x) \sin(\pi y)
  // u = u_0 y \sin(\pi x)
  // v = v_0 x \cos(\pi y)

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
  [[nodiscard]] real solution(const real x, const real y, const real time) const
      noexcept {
    return T_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real source_sol(const real x, const real y,
                                const real time) const noexcept {
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
  [[nodiscard]] real solution_dx(const real x, const real y,
                                 const real time) const noexcept {
    return pi * T_0() * std::sin(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy(const real x, const real y,
                                 const real time) const noexcept {
    return pi * T_0() * std::cos(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real solution_dx2(const real x, const real y,
                                  const real time) const noexcept {
    return -pi * pi * T_0() * std::cos(pi * x) * std::sin(pi * y);
  }

  [[nodiscard]] real solution_dy2(const real x, const real y,
                                  const real time) const noexcept {
    return solution_dx2(x, y, time);
  }

  [[nodiscard]] real initial_conds(const real x, const real y) const noexcept {
    return solution(x, y, 0.0);
  }

  [[nodiscard]] real boundary_x_min(const real y, const real time) const
      noexcept {
    return solution(x_min(), y, time);
  }

  [[nodiscard]] real boundary_x_max(const real y, const real time) const
      noexcept {
    return solution(x_max(), y, time);
  }

  [[nodiscard]] real boundary_y_min(const real x, const real time) const
      noexcept {
    return solution(x, y_min(), time);
  }

  [[nodiscard]] real boundary_y_max(const real x, const real time) const
      noexcept {
    return solution(x, y_max(), time);
  }

  [[nodiscard]] real boundary_dx_min(const real y, const real time) const
      noexcept {
    return solution_dx(x_min(), y, time);
  }

  [[nodiscard]] real boundary_dx_max(const real y, const real time) const
      noexcept {
    return solution_dx(x_min(), y, time);
  }

  [[nodiscard]] real boundary_dy_min(const real x, const real time) const
      noexcept {
    return solution_dy(x, y_min(), time);
  }

  [[nodiscard]] real boundary_dy_max(const real x, const real time) const
      noexcept {
    return solution_dy(x, y_max(), time);
  }

  constexpr BConds_Part1(const real T_0, const real u_0, const real v_0,
                         const real x_min = 0.0, const real x_max = 1.0,
                         const real y_min = 0.0, const real y_max = 1.0)
      : BConds_Base(T_0, u_0, v_0, x_min, x_max, y_min, y_max) {}

  constexpr BConds_Part1(const BConds_Base &src) : BConds_Base(src) {}
};

class BConds_Part5 : public BConds_Base {
 public:
  // T(x, y, 0) = y

  // T(x_min, y, t) = y + 3 / 4 * Pr * Ek * u_0 ** 2 * (1 - (1 - 2 * y) ** 4)

  // T(x_max, y, t) = 0

  // T(x, y_min, t) = 0

  // T(x, y_max, t) = 1

  // u(x, y) = 6 * u_0 * y * (1 - y)

  // v(x, y) = 0

  [[nodiscard]] real u(const real x, const real y) const noexcept {
    return 6.0 * u_0() * y * (1.0 - y);
  }

  [[nodiscard]] real v(const real x, const real y) const noexcept {
    return 0.0;
  }

  [[nodiscard]] real initial_conds(const real x, const real y) const noexcept {
    return y;
  }

  [[nodiscard]] real boundary_x_min(const real y, const real time) const
      noexcept {
    real pow_term = 1.0 - 2.0 * y;
    for(int i = 0; i < 2; i++) {
      pow_term *= pow_term;
    }
    return y + 3.0 / 4.0 * prandtl * eckert * u_0() * u_0() * (1.0 - pow_term);
  }

  [[nodiscard]] real boundary_x_max(const real y, const real time) const
      noexcept {
    return 0.0;
  }

  [[nodiscard]] real boundary_y_min(const real x, const real time) const
      noexcept {
    return 0.0;
  }

  [[nodiscard]] real boundary_y_max(const real x, const real time) const
      noexcept {
    return 1.0;
  }

  constexpr BConds_Part5(const real T_0, const real u_0, const real v_0,
                         const real x_min = 0.0, const real x_max = 1.0,
                         const real y_min = 0.0, const real y_max = 1.0)
      : BConds_Base(T_0, u_0, v_0, x_min, x_max, y_min, y_max) {}

  constexpr BConds_Part5(const BConds_Base &src) : BConds_Base(src) {}
};

#endif  // _BOUNDARIES_HPP_
