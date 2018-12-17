
#ifndef _BOUNDARIES_HPP_
#define _BOUNDARIES_HPP_

#include "constants.hpp"
#include "multivar.hpp"

#include <cmath>
#include <functional>

template <typename BConds_Impl>
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
  [[nodiscard]] constexpr real reynolds() const noexcept { return _reynolds; }

  constexpr BConds_Base(const real P_0, const real u_0, const real v_0,
                        const real beta, const real reynolds,
                        const real x_min = 0.0, const real x_max = 1.0,
                        const real y_min = 0.0, const real y_max = 1.0)
      : _p_0(P_0),
        _u_0(u_0),
        _v_0(v_0),
        _beta(beta),
        _reynolds(reynolds),
        _x_min(x_min),
        _x_max(x_max),
        _y_min(y_min),
        _y_max(y_max) {}

  constexpr BConds_Base(const BConds_Base &src)
      : BConds_Base(src.P_0(), src.u_0(), src.v_0(), src.beta(), src.reynolds(),
                    src.x_min(), src.x_max(), src.y_min(), src.y_max()) {}

  virtual ~BConds_Base() {}

  template <typename MeshT>
  void init_mesh(MeshT &mesh) const noexcept {
    for(int i = 0; i < mesh.x_dim(); i++) {
      for(int j = 0; j < mesh.y_dim(); j++) {
        const real x = mesh.x_median(i);
        const real y = mesh.y_median(j);
        mesh.press(i, j) =
            static_cast<const BConds_Impl *>(this)->pressure_initial(x, y);
        mesh.u_vel(i, j) =
            static_cast<const BConds_Impl *>(this)->u_vel_initial(x, y);
        mesh.v_vel(i, j) =
            static_cast<const BConds_Impl *>(this)->v_vel_initial(x, y);
      }
    }
  }

  triple initial_conds(const real x, const real y, const real dx,
                       const real dy) const noexcept {
    return {static_cast<const BConds_Impl *>(this)->pressure_initial(x, y),
            static_cast<const BConds_Impl *>(this)->u_vel_initial(x, y),
            static_cast<const BConds_Impl *>(this)->v_vel_initial(x, y)};
  };

 protected:
  real _p_0;
  real _u_0, _v_0;
  real _beta;
  real _reynolds;
  real _x_min, _x_max, _y_min, _y_max;
};

// dP/dt + (du/dx + dv/dy) / beta = 0.0
// du/dt + d(u^2)/dx + d(uv)/dy = (d^2u/dx^2 + d^2u/dy^2) / Re - dP/dx
// dv/dt + d(uv)/dx + d(v^2)/dy = (d^2v/dx^2 + d^2v/dy^2) / Re - dP/dy

class BConds_Part3 : public BConds_Base<BConds_Part3> {
 public:
  [[nodiscard]] real pressure_initial(const real x, const real y) const
      noexcept {
    return P_0() * std::cos(pi * x) * std::cos(pi * y);
  }

  [[nodiscard]] real u_vel_initial(const real x, const real y) const noexcept {
    return u_0() * std::sin(pi * x) * std::sin(2.0 * pi * y);
  }

  [[nodiscard]] real v_vel_initial(const real x, const real y) const noexcept {
    return v_0() * std::sin(2.0 * pi * x) * std::sin(pi * y);
  }

  template <typename MeshT>
  std::tuple<bool, int, int> boundary_coord(const MeshT &mesh, const int i,
                                            const int j) const noexcept {
    int i_edge      = i;
    int j_edge      = j;
    if(i == -1) {
      i_edge = 0;
    } else if(i == mesh.x_dim()) {
      i_edge -= 1;
    }
		if(j == -1) {
      j_edge = 0;
    } else if(j == mesh.y_dim()) {
      j_edge -= 1;
    }
		const bool ghost_cell = !(i == i_edge && j == j_edge);
    return {ghost_cell, i_edge, j_edge};
  }

  // Boundary Conditions:
  // At left and right boundaries
  // dP/dx = 0.0
  // u_left = u_right = 0.0
  // v_left = v_right = 0.0
  // At the bottom boundary
  // dP/dy = 0.0
  // u = 0.0
  // v = 0.0
  // At the top boundary
  // dP/dy = 0.0
  // u = u_wall
  // v = 0.0

  template <typename MeshT>
  [[nodiscard]] constexpr real pressure_at(const MeshT &mesh, const real time,
                                           const int i, const int j) const
      noexcept {
    const auto [ghost_cell, i_edge, j_edge] = boundary_coord(mesh, i, j);
    return mesh.press(i_edge, j_edge);
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real u_vel_at(const MeshT &mesh, const real time,
                                        const int i, const int j) const
      noexcept {
    const auto [ghost_cell, i_edge, j_edge] = boundary_coord(mesh, i, j);
    if(j == mesh.y_dim()) {
      return 2.0 * _wall_vel - mesh.u_vel(i_edge, j_edge);
    } else if(ghost_cell) {
      return -mesh.u_vel(i_edge, j_edge);
    } else {
      return mesh.u_vel(i, j);
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real v_vel_at(const MeshT &mesh, const real time,
                                        const int i, const int j) const
      noexcept {
    const auto [ghost_cell, i_edge, j_edge] = boundary_coord(mesh, i, j);
    if(ghost_cell) {
      return -mesh.v_vel(i_edge, j_edge);
    } else {
      return mesh.v_vel(i, j);
    }
  }

  constexpr BConds_Part3(const real wall_vel, const real P_0, const real u_0,
                         const real v_0, const real beta, const real reynolds,
                         const real x_min = 0.0, const real x_max = 1.0,
                         const real y_min = 0.0, const real y_max = 1.0)
      : BConds_Base(P_0, u_0, v_0, beta, reynolds, x_min, x_max, y_min, y_max),
        _wall_vel(wall_vel) {}

  constexpr BConds_Part3(const BConds_Part3 &src)
      : BConds_Base(src), _wall_vel(src._wall_vel) {}

 protected:
  real _wall_vel;
};

class BConds_Part1 : public BConds_Base<BConds_Part1> {
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
                       5.0 * pi * sx * s2y / reynolds()));
    const real v_term =
        pi * (P_0() * sy * cx -
              v_0() * (v_0() * s2y * s2x * s2x +
                       u_0() * sy * s2y * (cx * s2x + 2.0 * c2x * sx) +
                       5.0 * pi * sy * s2x / reynolds()));
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

  [[nodiscard]] real pressure_initial(const real x, const real y) const
      noexcept {
    return pressure(x, y, 0.0);
  }

  template <typename GetSol, typename GetMesh, typename MeshT>
  [[nodiscard]] constexpr real get_value_at(const GetSol &f, const GetMesh &g,
                                            const MeshT &mesh, const real time,
                                            const int i, const int j) const
      noexcept {
    if(i < 0 || i >= mesh.x_dim() || j < 0 || j >= mesh.y_dim()) {
      const real x = mesh.x_median(i);
      const real y = mesh.y_median(j);
      return f(x, y, time);
    } else {
      return g(mesh, i, j);
    }
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real pressure_at(const MeshT &mesh, const real time,
                                           const int i, const int j) const
      noexcept {
    return get_value_at([=](const real x, const real y,
                            const real t) { return pressure(x, y, t); },
                        [](const MeshT &mesh, const int i, const int j) {
                          return mesh.press(i, j);
                        },
                        mesh, time, i, j);
  }

  [[nodiscard]] real u_vel_initial(const real x, const real y) const noexcept {
    return u(x, y, 0.0);
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real u_vel_at(const MeshT &mesh, const real time,
                                        const int i, const int j) const
      noexcept {
    return get_value_at(
        [=](const real x, const real y, const real t) { return u(x, y, t); },
        [](const MeshT &mesh, const int i, const int j) {
          return mesh.u_vel(i, j);
        },
        mesh, time, i, j);
  }

  [[nodiscard]] real v_vel_initial(const real x, const real y) const noexcept {
    return v(x, y, 0.0);
  }

  template <typename MeshT>
  [[nodiscard]] constexpr real v_vel_at(const MeshT &mesh, const real time,
                                        const int i, const int j) const
      noexcept {
    return get_value_at(
        [=](const real x, const real y, const real t) { return v(x, y, t); },
        [](const MeshT &mesh, const int i, const int j) {
          return mesh.v_vel(i, j);
        },
        mesh, time, i, j);
  }

  constexpr BConds_Part1(const real P_0, const real u_0, const real v_0,
                         const real beta, const real reynolds,
                         const real x_min = 0.0, const real x_max = 1.0,
                         const real y_min = 0.0, const real y_max = 1.0)
      : BConds_Base(P_0, u_0, v_0, beta, reynolds, x_min, x_max, y_min, y_max) {
  }

  constexpr BConds_Part1(const BConds_Base &src) : BConds_Base(src) {}
};

#endif  // _BOUNDARIES_HPP_
