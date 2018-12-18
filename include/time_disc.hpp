
#ifndef _TIME_DISC_HPP_
#define _TIME_DISC_HPP_

#include "boundaries.hpp"
#include "constants.hpp"
#include "space_disc.hpp"
#include "thomas.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>

template <typename _Mesh, typename _SpaceDisc>
class Base_Solver {
 public:
  using MeshT         = _Mesh;
  using SpaceAssembly = INSAssembly<_SpaceDisc>;
  using BConds        = typename SpaceAssembly::BConds;

  Base_Solver(const BConds &boundaries)
      : _cur_mesh(std::make_unique<MeshT>(boundaries)),
        _space_assembly(boundaries),
        _time(0.0) {
    boundaries.init_mesh(*_cur_mesh);
  }

  [[nodiscard]] constexpr real time() const noexcept { return _time; }

  [[nodiscard]] constexpr const MeshT &mesh() const noexcept {
    return *_cur_mesh;
  }

  [[nodiscard]] constexpr MeshT &mesh() noexcept { return *_cur_mesh; }

  [[nodiscard]] constexpr SpaceAssembly &space_assembly() noexcept {
    return _space_assembly;
  }

  [[nodiscard]] constexpr const BConds &boundaries() const noexcept {
    return _space_assembly.boundaries();
  }

 protected:
  std::unique_ptr<MeshT> _cur_mesh;
  SpaceAssembly _space_assembly;
  real _time;
};

template <typename _Mesh, typename _SpaceDisc>
class ImplicitEuler_Solver : public Base_Solver<_Mesh, _SpaceDisc> {
 public:
  using MeshT         = _Mesh;
  using SpaceAssembly = INSAssembly<_SpaceDisc>;
  using Base          = Base_Solver<MeshT, _SpaceDisc>;
  using BConds        = typename Base::BConds;

  // Definitions used to solve the system

  // Assuming a second order space discretization...
  static constexpr int ghost_cells = 2;

  static constexpr int vec_x_dim = MeshT::x_dim() + ghost_cells;
  static constexpr int vec_y_dim = MeshT::y_dim() + ghost_cells;

  using SolVecX = ND_Array<triple, vec_x_dim>;
  using SolVecY = ND_Array<triple, vec_y_dim>;

  using XYIntermediate = ND_Array<triple, MeshT::x_dim(), MeshT::y_dim()>;

  using MtxX = ND_Array<Jacobian, vec_x_dim, 3>;
  using MtxY = ND_Array<Jacobian, vec_y_dim, 3>;

  constexpr ImplicitEuler_Solver(const BConds &boundaries) noexcept
      : Base(boundaries),
        _dx(std::make_unique<MtxX>()),
        _dy(std::make_unique<MtxY>()),
        _sol_dx(std::make_unique<SolVecX>()),
        _sol_dy(std::make_unique<SolVecY>()),
        _intermediate(std::make_unique<XYIntermediate>()) {}

  real timestep(const real dt) {
    const MeshT &mesh                   = this->mesh();
    const SpaceAssembly &space_assembly = this->space_assembly();
    // First we need to construct our vector to use with the Thomas algorithm
    // Start with the x direction - we need a ghost cell on each vertical edge
    // In the y direction we need a ghost cell on each horizontal edge
    // We also want the matrix to be tridiagonal, so we need to look at the
    // transpose of the original mesh

    // Fill the matrix with the Dx Terms
    // Fill the solution vector with the flux integral
    // Then solve for the solution to our y vector
    // Then fill the matrix with the Dy Terms
    // And solve for the dU vector
    //
    // Recall that (after approximate factorization) the system we're solving
    // looks like so:
    //
    // [ I     I     0     0         ... ] [ dU_{0, 1} ]   [ -dt FI_{0, 1} ]
    // [ Dx-1  Dx0   Dx1   0         ... ] [ dU_{1, 1} ]   [ -dt FI_{1, 1} ]
    // [ 0     Dx-1  Dx0   Dx1       ... ] [ dU_{2, 1} ]   [ -dt FI_{2, 1} ]
    // [ 0     0     Dx-1  Dx0       ... ] [ dU_{3, 1} ]   [ -dt FI_{3, 1} ]
    // [ .           .                .  ] [     .     ]   [       .       ]
    // [ .                 .          .  ] [     .     ]   [       .       ]
    // [ .                     .      .  ] [     .     ]   [       .       ]
    // [ ...   I     I     0     0   ... ] [ dU_{0, 2} ] = [ -dt FI_{0, 2} ]
    // [ ...   Dx-1  Dx0   Dx1   0   ... ] [ dU_{1, 2} ]   [ -dt FI_{1, 2} ]
    // [ ...   0     Dx-1  Dx0   Dx1 ... ] [ dU_{2, 2} ]   [ -dt FI_{2, 2} ]
    // [ ...   0     0     Dx-1  Dx0 ... ] [ dU_{3, 2} ]   [ -dt FI_{3, 2} ]
    // [ ...   .     .     .             ] [     .     ]   [       .       ]
    // [ ...   .     .           .       ] [     .     ]   [       .       ]
    // [ ...   .     .               .   ] [     .     ]   [       .       ]
    //
    // This is an M*N x M*N matrix, where M = x_ctrl_vols + 2, N = y_ctrl_vols
    // Since each strip of j and i values is completely independent after
    // approximate factorization, we only consider them individually
    //
    SolVecX &sol_dx = *_sol_dx;
    SolVecY &sol_dy = *_sol_dy;

    MtxX &dx = *_dx;

    constexpr real snan = std::numeric_limits<real>::signaling_NaN();

    for(int i = 0; i < MeshT::x_dim(); i++) {
      for(int j = 0; j < MeshT::x_dim(); j++) {
        (*_intermediate)(i, j) = {snan, snan, snan};
      }
    }

    // Start in the X direction, with constant j
    for(int j = 0; j < MeshT::y_dim(); j++) {
      // Enforce our boundary conditions
      // All of our boundary conditions require the walls to be no slip
      sol_dx(0)             = 0.0;
      sol_dx(vec_x_dim - 1) = 0.0;

      dx(0, 0)       = Jacobian(Jacobian::NaNTag());  // Should never be used
      dx(0, 1)       = Jacobian(Jacobian::IdentityTag());
      dx(0, 2)       = Jacobian(Jacobian::IdentityTag());
      dx(0, 1)(0, 0) = -1.0;

      dx(vec_x_dim - 1, 0) = Jacobian(Jacobian::IdentityTag());
      dx(vec_x_dim - 1, 1) = Jacobian(Jacobian::IdentityTag());
      dx(vec_x_dim - 1, 2) =
          Jacobian(Jacobian::NaNTag());  // Should never be used
      dx(vec_x_dim - 1, 1)(0, 0) = -1.0;

      for(int i = 0; i < MeshT::x_dim(); i++) {
        // We need to offset the x values for the ghost cells
        sol_dx(i + 1) =
            space_assembly.flux_integral(mesh, i, j, this->time()) * dt;

        dx(i + 1, 0) = space_assembly.Dx_m1(mesh, i, j, this->time()) * dt;
        dx(i + 1, 1) = Jacobian(Jacobian::IdentityTag()) +
                       space_assembly.Dx_0(mesh, i, j, this->time()) * dt;
        dx(i + 1, 2) = space_assembly.Dx_p1(mesh, i, j, this->time()) * dt;
      }

      solve_thomas(dx, sol_dx);

      XYIntermediate &inter = *_intermediate;
      for(int i = 0; i < MeshT::x_dim(); i++) {
        // Skip the ghost cell in the solution since it's not needed
        inter(i, j) = sol_dx(i + 1);
      }
    }

    for(int i = 0; i < MeshT::x_dim(); i++) {
      MtxY &dy = *_dy;
      // Enforce our boundary conditions
      dy(0, 0)       = Jacobian(Jacobian::NaNTag());
      dy(0, 1)       = Jacobian(Jacobian::IdentityTag());
      dy(0, 2)       = Jacobian(Jacobian::IdentityTag());
      dy(0, 1)(0, 0) = -1.0;

      dy(vec_y_dim - 1, 0)       = Jacobian(Jacobian::IdentityTag());
      dy(vec_y_dim - 1, 1)       = Jacobian(Jacobian::IdentityTag());
      dy(vec_y_dim - 1, 2)       = Jacobian(Jacobian::NaNTag());
      dy(vec_y_dim - 1, 1)(0, 0) = -1.0;

      sol_dy(0)                  = 0.0;
      sol_dy(MeshT::y_dim() + 1) = 0.0;

      XYIntermediate &inter = *_intermediate;
      for(int j = 0; j < MeshT::y_dim(); j++) {
        sol_dy(j + 1) = inter(i, j);

        dy(j + 1, 0) = space_assembly.Dy_m1(mesh, i, j, this->time()) * dt;
        dy(j + 1, 1) = Jacobian(Jacobian::IdentityTag()) +
                       space_assembly.Dy_0(mesh, i, j, this->time()) * dt;
        dy(j + 1, 2) = space_assembly.Dy_p1(mesh, i, j, this->time()) * dt;
      }

      solve_thomas(dy, sol_dy);

      for(int j = 0; j < MeshT::y_dim(); j++) {
        inter(i, j) = sol_dy(j + 1);
      }
    }

    real max_change = 0.0;

    // sol now contains our dP, du, and dv terms
    // So just add it to our P, u, and v terms
    // Recall that our sol_mesh is transposed, so we need to swap the indices
    // used for it
    const XYIntermediate &inter = *_intermediate;
    for(int i = 0; i < MeshT::x_dim(); i++) {
      for(int j = 0; j < MeshT::y_dim(); j++) {
        max_change = std::max(
            max_change, this->boundaries().relax() * inter(i, j).l2_norm());
        this->_cur_mesh->press(i, j) +=
            inter(i, j)(0) * this->boundaries().relax();
        this->_cur_mesh->u_vel(i, j) +=
            inter(i, j)(1) * this->boundaries().relax();
        this->_cur_mesh->v_vel(i, j) +=
            inter(i, j)(2) * this->boundaries().relax();
      }
    }

    this->_time += dt;
    return max_change;
  }

 private:
  // These are typically too large to keep on the stack (well, without 'ulimit
  // -s unlimited', but we'd rather not rely on that...), so allocate them here
  std::unique_ptr<MtxX> _dx;
  std::unique_ptr<MtxY> _dy;
  std::unique_ptr<SolVecX> _sol_dx;
  std::unique_ptr<SolVecY> _sol_dy;
  std::unique_ptr<XYIntermediate> _intermediate;
};

template <typename _Mesh, typename _SpaceDisc>
class RK1_Solver : public Base_Solver<_Mesh, _SpaceDisc> {
 public:
  using MeshT         = _Mesh;
  using SpaceAssembly = INSAssembly<_SpaceDisc>;
  using Base          = Base_Solver<MeshT, _SpaceDisc>;
  using BConds        = typename Base::BConds;

  constexpr RK1_Solver(const BConds &boundaries) noexcept
      : Base(boundaries), _partial_mesh(std::make_unique<MeshT>(boundaries)) {}

  real timestep(const real sigma_ratio) {
    MeshT &mesh = this->mesh();
    // |sigma| = |1 + \lambda \Delta t|
    // Approximate \lambda with just the second derivative term,
    // since it will dominate with small \Delta x.
    // Then |\sigma| \leq 1 + (4 / (Re * Pr)(\Delta t / \Delta x^2) = 1
    // So our maximum timestep is \Delta x^2 * Re * Pr / 4

    const real dt = sigma_ratio * mesh.dx() * mesh.dx() *
                    this->boundaries().reynolds() * prandtl / 4.0;

    const real max_delta = this->_space_assembly.flux_assembly(
        *(this->_cur_mesh), *(this->_cur_mesh), *_partial_mesh, this->time(),
        dt);

    std::swap(this->_cur_mesh, _partial_mesh);
    this->_time += dt;
    return max_delta;
  }

 protected:
  // Note that storing the mesh as a pointer here shouldn't affect performance -
  // the number of dereferences should be the same
  std::unique_ptr<MeshT> _partial_mesh;
};

template <typename _Mesh, typename _SpaceAssembly>
class RK4_Solver : public Base_Solver<_Mesh, _SpaceAssembly> {
 public:
  using MeshT         = _Mesh;
  using SpaceAssembly = _SpaceAssembly;
  using Base          = Base_Solver<MeshT, SpaceAssembly>;
  using BConds        = typename Base::BConds;

  constexpr RK4_Solver(const BConds &boundaries) noexcept
      : Base(boundaries),
        _partial_mesh_1(std::make_unique<MeshT>(boundaries)),
        _partial_mesh_2(std::make_unique<MeshT>(boundaries)) {}

  real timestep(const real sigma_ratio) {
    MeshT &mesh = this->mesh();
    // Approximate the amplification factor as
    // 4^3 (\Delta t / \Delta x^2)^4 / (6 * Re * Pr) = sigma_mag
    // where sigma_mag = sigma_ratio^4
    // We can make this approximation because for small \Delta x,
    // the other terms of the amplification factor will be small in comparison
    // Then solve for \Delta t
    const real constants = std::sqrt(
        std::sqrt(6.0 * this->boundaries().reynolds() * prandtl / 64.0));
    const real ds = std::max(mesh.dx(), mesh.dy());
    const real dt = sigma_ratio * constants * (ds * ds);

    this->_space_assembly.flux_assembly(mesh, mesh, *_partial_mesh_1,
                                        this->time(), dt / 4.0);

    // Second stage
    // compute w(2) based on w(1) and the current timestep
    this->_space_assembly.flux_assembly(mesh, *_partial_mesh_1,
                                        *_partial_mesh_2,
                                        this->time() + dt / 2.0, dt / 3.0);

    // Third stage
    this->_space_assembly.flux_assembly(mesh, *_partial_mesh_2,
                                        *_partial_mesh_1,
                                        this->time() + dt / 2.0, dt / 2.0);

    // Fourth stage
    this->_space_assembly.flux_assembly(
        mesh, *_partial_mesh_1, *_partial_mesh_2, this->time() + dt / 2.0, dt);

    real max_delta = 0.0;
    for(int i = 0; i < mesh.x_dim(); i++) {
      for(int j = 0; j < mesh.y_dim(); j++) {
        const triple delta = (*_partial_mesh_2)(i, j) - mesh(i, j);
        max_delta          = std::max(max_delta, delta.l2_norm());
      }
    }

    std::swap(this->_cur_mesh, _partial_mesh_2);
    this->_time += dt;
    return max_delta;
  }

 protected:
  std::unique_ptr<MeshT> _partial_mesh_1;
  std::unique_ptr<MeshT> _partial_mesh_2;
};

#endif  // _TIME_DISC_HPP_
