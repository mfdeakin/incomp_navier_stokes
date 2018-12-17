
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

  // Assuming a second order space discretization...
  static constexpr int ghost_cells = 2;

  // Definitions used to solve the system
  using VecMeshX =
      ND_Array<triple, MeshT::y_dim(), MeshT::x_dim() + ghost_cells>;
  using VecMeshY =
      ND_Array<triple, MeshT::x_dim(), MeshT::y_dim() + ghost_cells>;

  static constexpr int vec_x_dim = VecMeshX::extent(0) * VecMeshX::extent(1);
  static constexpr int vec_y_dim = VecMeshY::extent(0) * VecMeshY::extent(1);

  using SolVecX = ND_Array<triple, vec_x_dim>;
  using SolVecY = ND_Array<triple, vec_y_dim>;

  using MtxX = ND_Array<Jacobian, vec_x_dim, 3>;
  using MtxY = ND_Array<Jacobian, vec_y_dim, 3>;
  using MtxXMesh =
      ND_Array<Jacobian, VecMeshX::extent(0), VecMeshX::extent(1), 3>;
  using MtxYMesh =
      ND_Array<Jacobian, VecMeshY::extent(0), VecMeshY::extent(1), 3>;

  constexpr ImplicitEuler_Solver(const BConds &boundaries) noexcept
      : Base(boundaries),
        _dx(std::make_unique<MtxX>()),
        _dy(std::make_unique<MtxY>()),
        _sol_dx(std::make_unique<SolVecX>()),
        _sol_dy(std::make_unique<SolVecY>()) {}

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
    //
    SolVecX &sol_dx       = *_sol_dx;
    VecMeshX &sol_dx_mesh = sol_dx.template reshape<VecMeshX>();

    MtxX &dx          = *_dx;
    MtxXMesh &dx_mesh = dx.template reshape<MtxXMesh>();

    // Enforce our boundary conditions
    // In the X direction,
    for(int j = 0; j < MeshT::y_dim(); j++) {
      // All of our boundary conditions require the walls to be non-porous
      sol_dx_mesh(j, 0)                  = 0.0;
      sol_dx_mesh(j, MeshT::x_dim() + 1) = 0.0;

      dx_mesh(j, 0, 0) = Jacobian(Jacobian::ZeroTag());
      dx_mesh(j, 0, 1) = Jacobian(Jacobian::IdentityTag());
      dx_mesh(j, 0, 2) = Jacobian(Jacobian::IdentityTag());

      dx_mesh(j, MtxXMesh::extent(1) - 1, 0) =
          Jacobian(Jacobian::IdentityTag());
      dx_mesh(j, MtxXMesh::extent(1) - 1, 1) =
          Jacobian(Jacobian::IdentityTag());
      dx_mesh(j, MtxXMesh::extent(1) - 1, 2) = Jacobian(Jacobian::ZeroTag());
    }
    for(int j = 0; j < MeshT::y_dim(); j++) {
      for(int i = 0; i < MeshT::x_dim(); i++) {
        // We need to offset the x values for the ghost cells
        sol_dx_mesh(j, i + 1) =
            space_assembly.flux_integral(mesh, i, j, this->time()) * dt;

        dx_mesh(j, i + 1, 0) =
            space_assembly.Dx_m1(mesh, i, j, this->time()) * dt;
        dx_mesh(j, i + 1, 1) =
            Jacobian(Jacobian::IdentityTag()) +
            space_assembly.Dx_0(mesh, i, j, this->time()) * dt;
        dx_mesh(j, i + 1, 2) =
            space_assembly.Dx_p1(mesh, i, j, this->time()) * dt;
      }
    }

    // Up to here, sol_dx is correct and solve_thomas is correct
    // Need to understand why dx is wrong

    solve_thomas(dx, sol_dx);
    MtxY &dy          = *_dy;
    MtxYMesh &dy_mesh = dy.template reshape<MtxYMesh>();

    SolVecY &sol_dy       = *_sol_dy;
    VecMeshY &sol_dy_mesh = sol_dy.template reshape<VecMeshY>();
    // Enforce our boundary conditions
    for(int i = 0; i < MeshT::x_dim(); i++) {
      dy_mesh(i, 0, 0) = Jacobian(Jacobian::ZeroTag());
      dy_mesh(i, 0, 1) = Jacobian(Jacobian::IdentityTag());
      dy_mesh(i, 0, 2) = Jacobian(Jacobian::IdentityTag());

      dy_mesh(i, MtxYMesh::extent(1) - 1, 0) =
          Jacobian(Jacobian::IdentityTag());
      dy_mesh(i, MtxYMesh::extent(1) - 1, 1) =
          Jacobian(Jacobian::IdentityTag());
      dy_mesh(i, MtxYMesh::extent(1) - 1, 2) = Jacobian(Jacobian::ZeroTag());

      sol_dy_mesh(i, 0)                  = 0.0;
      sol_dy_mesh(i, MeshT::y_dim() + 1) = 0.0;
    }

    for(int i = 0; i < MeshT::x_dim(); i++) {
      for(int j = 0; j < MeshT::y_dim(); j++) {
        // Note that since the Thomas algorithm solves a tridiagonal system,
        // we need to swap some rows of the solution to make the matrix
        // tridiagonal ie, go from increasing the x index to increasing the y
        // index. This is easiest to achieve by taking the transpose of the
        // solution when looking at it like a matrix
        // We also ignore the ghost cells, since those are degenerate between
        // directions
        sol_dy_mesh(i, j + 1) = sol_dx_mesh(j, i + 1);
      }
    }

    for(int i = 0; i < MeshT::x_dim(); i++) {
      for(int j = 0; j < MeshT::y_dim(); j++) {
        dy_mesh(i, j + 1, 0) =
            space_assembly.Dy_m1(mesh, i, j, this->time()) * dt;
        dy_mesh(i, j + 1, 1) =
            Jacobian(Jacobian::IdentityTag()) +
            space_assembly.Dy_0(mesh, i, j, this->time()) * dt;
        dy_mesh(i, j + 1, 2) =
            space_assembly.Dy_p1(mesh, i, j, this->time()) * dt;
      }
    }

    solve_thomas(dy, sol_dy);

    real max_change = 0.0;

    // sol now contains our dP, du, and dv terms
    // So just add it to our P, u, and v terms
    // Recall that our sol_mesh is transposed, so we need to swap the indices
    // used for it
    for(int i = 0; i < MeshT::x_dim(); i++) {
      for(int j = 0; j < MeshT::y_dim(); j++) {
        max_change = std::max(max_change, sol_dy_mesh(i, j + 1).l2_norm());
        this->_cur_mesh->press(i, j) += sol_dy_mesh(i, j + 1)(0);
        this->_cur_mesh->u_vel(i, j) += sol_dy_mesh(i, j + 1)(1);
        this->_cur_mesh->v_vel(i, j) += sol_dy_mesh(i, j + 1)(2);
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
