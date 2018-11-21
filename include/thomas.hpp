
#ifndef _THOMAS_HPP_
#define _THOMAS_HPP_

#include "constants.hpp"

#include <limits>

/* Solve a tri-diagonal system Ax = b.
 *
 * Uses the Thomas algorithm, which is Gauss elimination and back
 * substitution specialized for a tri-diagonal matrix.
 *
 * Input:
 *  LHS: The matrix A.  The three columns are the three non-zero diagonals.
 *     (0: below main diag; 1: on main diag; 2: above main diag)
 *  RHS: A vector containing b.
 *
 * Output:
 *  LHS: Garbled.
 *  RHS: The solution x.
 */
template <typename Mtx>
constexpr void solve_thomas(Mtx &LHS,
                            ND_Array<real, Mtx::extent(0)> &RHS) noexcept {
  constexpr int ctrl_vols = Mtx::extent(0);
  /* This next line actually has no effect, but it -does- make clear that
     the values in those locations have no impact. */
  LHS(0, 0) = LHS(ctrl_vols - 1, 2) = std::numeric_limits<real>::quiet_NaN();
  /* Forward elimination */
  for(int i = 0; i < ctrl_vols - 1; i++) {
    LHS(i, 2) /= LHS(i, 1);
    RHS(i) /= LHS(i, 1);
    LHS(i + 1, 1) -= LHS(i, 2) * LHS(i + 1, 0);
    RHS(i + 1) -= LHS(i + 1, 0) * RHS(i);
  }
  /* Last line of elimination */
  RHS(ctrl_vols - 1) /= LHS(ctrl_vols - 1, 1);

  /* Back-substitution */
  for(int i = ctrl_vols - 2; i >= 0; i--) {
    RHS(i) -= RHS(i + 1) * LHS(i, 2);
  }
}

#endif  // _THOMAS_HPP_
