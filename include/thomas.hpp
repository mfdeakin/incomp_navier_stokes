
#ifndef _THOMAS_HPP_
#define _THOMAS_HPP_

#include "constants.hpp"

#include <limits>

/* Author: Carl Ollivier-Gooch
 * Solve a tri-diagonal system Ax = b.
 *
 * Uses the Thomas algorithm, which is Gauss elimination and back
 * substitution specialized for a tri-diagonal matrix.
 *
 * Input:
 *  lhs: The matrix A.  The three columns are the three non-zero diagonals.
 *     (0: below main diag; 1: on main diag; 2: above main diag)
 *  rhs: A vector containing b.
 *
 * Output:
 *  lhs: Garbled.
 *  rhs: The solution x.
 */
template <typename Mtx>
constexpr void solve_thomas(Mtx &lhs,
                            ND_Array<real, Mtx::extent(0)> &rhs) noexcept {
  constexpr int ctrl_vols = Mtx::extent(0);
  /* This next line actually has no effect, but it -does- make clear that
     the values in those locations have no impact. */
  lhs(0, 0) = lhs(ctrl_vols - 1, 2) = std::numeric_limits<real>::quiet_NaN();
  /* Forward elimination */
  for(int i = 0; i < ctrl_vols - 1; i++) {
    lhs(i, 2) /= lhs(i, 1);
    rhs(i) /= lhs(i, 1);
    lhs(i + 1, 1) -= lhs(i, 2) * lhs(i + 1, 0);
    rhs(i + 1) -= lhs(i + 1, 0) * rhs(i);
  }
  /* Last line of elimination */
  rhs(ctrl_vols - 1) /= lhs(ctrl_vols - 1, 1);

  /* Back-substitution */
  for(int i = ctrl_vols - 2; i >= 0; i--) {
    rhs(i) -= rhs(i + 1) * lhs(i, 2);
  }
}

#endif  // _THOMAS_HPP_
