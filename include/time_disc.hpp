
#ifndef _TIME_DISC_HPP_
#define _TIME_DISC_HPP_

#include "constants.hpp"
#include "thomas.hpp"

#include <algorithm>
#include <memory>

template <typename _Mesh, typename _SpaceAssembly>
class Base_Solver {
 public:
  using MeshT         = _Mesh;
  using SpaceAssembly = _SpaceAssembly;

  Base_Solver(const real T_0 = 1.0, const real u_0 = 1.0, const real v_0 = 1.0,
              const real x_min = 0.0, const real x_max = 1.0,
              const real y_min = 0.0, const real y_max = 1.0)
      : _cur_mesh(std::make_unique<MeshT>(x_min, x_max, y_min, y_max)),
        _space_assembly(T_0, u_0, v_0),
        _time(0.0) {
    for(int i = 0; i < _cur_mesh->x_dim(); i++) {
      for(int j = 0; j < _cur_mesh->y_dim(); j++) {
        const real x = _cur_mesh->x_median(i);
        const real y = _cur_mesh->y_median(j);

        _cur_mesh->Temp(i, j)  = _space_assembly.solution(x, y);
        _cur_mesh->u_vel(i, j) = _space_assembly.u(x, y);
        _cur_mesh->v_vel(i, j) = _space_assembly.v(x, y);
      }
    }
  }

  [[nodiscard]] constexpr real time() const noexcept { return _time; }

  [[nodiscard]] constexpr MeshT &mesh() const noexcept { return *_cur_mesh; }

  [[nodiscard]] constexpr SpaceAssembly &space_assembly() noexcept {
    return _space_assembly;
  }

 protected:
  std::unique_ptr<MeshT> _cur_mesh;
  SpaceAssembly _space_assembly;
  real _time;
};

template <typename _Mesh, typename _SpaceAssembly>
class ImplicitEuler_Solver : public Base_Solver<_Mesh, _SpaceAssembly> {
 public:
  using MeshT         = _Mesh;
  using SpaceAssembly = _SpaceAssembly;

  ImplicitEuler_Solver(const real T_0 = 1.0, const real u_0 = 1.0,
                       const real v_0 = 1.0, const real x_min = 0.0,
                       const real x_max = 1.0, const real y_min = 0.0,
                       const real y_max = 1.0)
      : Base_Solver<_Mesh, _SpaceAssembly>(T_0, u_0, v_0, x_min, x_max, y_min,
                                           y_max) {}

  void timestep(const real dt) {
    // First we need to construct our vector to use with the Thomas algorithm
    constexpr int vec_dim = (MeshT::x_dim()) * (MeshT::y_dim());
    using VecT            = ND_Array<real, vec_dim>;
    using MtxT            = ND_Array<real, vec_dim, 3>;

    // Fill MtxT with our Dx Terms
    // Fill VecT with the source terms and the flux integral
    // Then solve for the solution to our y vector
    // Then fill MtxT with our Dy Terms
    // And solve for our dT vector
    typename MeshT::ControlVolumes sol_mesh;
    VecT &sol = sol_mesh.reshape();
    MtxT dx;

    for(int i = 0, m = 0; i < MeshT::x_dim(); i++) {
      for(int j = 0; j < MeshT::y_dim(); j++, m++) {
        sol_mesh(i, j) =
            (this->_space_assembly.flux_integral(*(this->_cur_mesh), i, j) +
             this->space_assembly.source_fd(*(this->_cur_mesh), i, j)) *
            dt;
        dx(m, 0) = 0.0;
        dx(m, 1) = 0.0;
        dx(m, 2) = 0.0;
      }
    }
    solve_thomas(dx, sol);

    MtxT dy;
    for(int i = 0, m = 0; i < MeshT::x_dim(); i++) {
      for(int j = 0; j < MeshT::y_dim(); j++, m++) {
        dy(m, 0) = 0.0;
        dy(m, 1) = 0.0;
        dy(m, 2) = 0.0;
      }
    }
    solve_thomas(dy, sol);

    // sol now contains our dT terms
    // So just add it to our T terms
    for(int i = 0, m = 0; i < MeshT::x_dim(); i++) {
      for(int j = 0; j < MeshT::y_dim(); j++, m++) {
        this->_cur_mesh->Temp(i, j) += sol_mesh(i, j);
      }
    }
  }
};

template <typename _Mesh, typename _SpaceAssembly>
class RK4_Solver : public Base_Solver<_Mesh, _SpaceAssembly> {
 public:
  using MeshT         = _Mesh;
  using SpaceAssembly = _SpaceAssembly;

  RK4_Solver(const real T_0 = 1.0, const real u_0 = 1.0, const real v_0 = 1.0,
             const real x_min = 0.0, const real x_max = 1.0,
             const real y_min = 0.0, const real y_max = 1.0)
      : _partial_mesh_1(std::make_unique<MeshT>(x_min, x_max, y_min, y_max)),
        _partial_mesh_2(std::make_unique<MeshT>(x_min, x_max, y_min, y_max)) {}

  void timestep(const real cfl) {
    const real dt =
        cfl * this->_cur_mesh->dx() /
        std::max(this->_space_assembly.u_0(), this->_space_assembly.v_0());

    this->_space_assembly.flux_assembly(*(this->_cur_mesh), *(this->_cur_mesh),
                                        *_partial_mesh_1, this->time(),
                                        dt / 4.0);

    // Second stage
    // compute w(2) based on w(1) and the current timestep
    this->_space_assembly.flux_assembly(*(this->_cur_mesh), *_partial_mesh_1,
                                        *_partial_mesh_2,
                                        this->time() + dt / 2.0, dt / 3.0);

    // Third stage
    this->_space_assembly.flux_assembly(*(this->_cur_mesh), *_partial_mesh_2,
                                        *_partial_mesh_1,
                                        this->time() + dt / 2.0, dt / 2.0);

    // Fourth stage
    this->_space_assembly.flux_assembly(*(this->_cur_mesh), *_partial_mesh_1,
                                        *_partial_mesh_2,
                                        this->time() + dt / 2.0, dt);

    std::swap(this->_cur_mesh, _partial_mesh_2);
    this->_time += dt;
  }

 protected:
  std::unique_ptr<MeshT> _partial_mesh_1;
  std::unique_ptr<MeshT> _partial_mesh_2;
};

#endif  // _TIME_DISC_HPP_
