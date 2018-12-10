
#ifndef _THOMAS_HPP_
#define _THOMAS_HPP_

#include "constants.hpp"
#include "nd_array/nd_array.hpp"

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
template <int ctrl_vols>
constexpr void solve_thomas(ND_Array<Jacobian, ctrl_vols, 3> &lhs,
                            ND_Array<triple, ctrl_vols> &rhs) noexcept {
  /* Forward elimination */
  for(int i = 0; i < ctrl_vols - 1; i++) {
    // Since we're dealing with non-comutative objects (matrices) in our matrix,
    // we need to be careful to always multiply on the left, since that's within
    // our powers, rather than the right
    // Using the inverse probably isn't the fastest way of doing this,
    // but it's easy and pretty, and for a 3x3 matrix, not too expensive
    const Jacobian inv = lhs(i, 1).inverse();

    rhs(i) = inv * rhs(i);
    // At this point, treat the following as true
    // lhs(i, 0) = Jacobian(Jacobian::ZeroTag());
    // lhs(i, 1) = Jacobian(Jacobian::IdentityTag());
    lhs(i, 2) = inv * lhs(i, 2);
    lhs(i + 1, 1) -= lhs(i + 1, 0) * lhs(i, 2);
    rhs(i + 1) -= lhs(i + 1, 0) * rhs(i);
  }
  /* Last line of elimination */
  rhs(ctrl_vols - 1) = lhs(ctrl_vols - 1, 1).inverse() * rhs(ctrl_vols - 1);

  /* Back-substitution */
  for(int i = ctrl_vols - 2; i >= 0; i--) {
    rhs(i) -= lhs(i, 2) * rhs(i + 1);
  }
}

#endif  // _THOMAS_HPP_
