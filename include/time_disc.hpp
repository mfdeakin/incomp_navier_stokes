
#ifndef _SOLVER_HPP_
#define _SOLVER_HPP_

#include "constants.hpp"

#include <algorithm>
#include <memory>

template <typename _MeshT, typename _SpaceAssemblyT>
class RK4_Solver {
 public:
  using MeshT          = _MeshT;
  using SpaceAssemblyT = _SpaceAssemblyT;

  RK4_Solver(const real T_0 = 1.0, const real u_0 = 1.0, const real v_0 = 1.0,
             const real x_min = 0.0, const real x_max = 1.0,
             const real y_min = 0.0, const real y_max = 1.0)
      : _cur_mesh(std::make_unique(x_min, x_max, y_min, y_max)),
        _partial_mesh_1(std::make_unique(x_min, x_max, y_min, y_max)),
        _partial_mesh_2(std::make_unique(x_min, x_max, y_min, y_max)),
        _space_assembly(T_0, u_0, v_0),
        _time(0.0) {}

  void timestep(const real cfl) {
    const real dt =
        cfl * _cur_mesh->dx() / std::max(_space_assembly.u0(), _space_assembly.v0());

    _space_assembly.flux_assembly(*_cur_mesh, *_cur_mesh, *_partial_mesh_1, time(),
                              dt() / 4.0);

    // Second stage
    // compute w(2) based on w(1) and the current timestep
    _space_assembly.flux_assembly(*_cur_mesh, *_partial_mesh_1, *_partial_mesh_2,
                              time() + dt() / 2.0, dt() / 3.0);

    // Third stage
    _space_assembly.flux_assembly(*_cur_mesh, *_partial_mesh_2, *_partial_mesh_1,
                              time() + dt() / 2.0, dt() / 2.0);

    // Fourth stage
    _space_assembly.flux_assembly(*_cur_mesh, *_partial_mesh_1, *_partial_mesh_2,
                              time() + dt() / 2.0, dt());

    std::swap(_cur_mesh, _partial_mesh_2);
    _time += dt();
  }

 protected:
  std::unique_ptr<MeshT> _cur_mesh;
  std::unique_ptr<MeshT> _partial_mesh_1;
  std::unique_ptr<MeshT> _partial_mesh_2;

  SpaceAssemblyT _space_assembly;

	real _time;
};

#endif  // _SOLVER_HPP_
