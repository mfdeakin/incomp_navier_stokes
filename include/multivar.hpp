
#ifndef _MULTIVAR_HPP_
#define _MULTIVAR_HPP_

#include "constants.hpp"

#include "nd_array/nd_array.hpp"

#include <cstddef>
#include <initializer_list>
#include <utility>

class triple : public ND_Array<real, 3> {
 public:
  triple() : ND_Array<real, 3>() {}

  // Make triples compatible with structured bindings and initializer lists
  triple(std::initializer_list<real> items) : ND_Array<real, 3>() {
    int idx = 0;
    for(real v : items) {
      (*this)(idx) = v;
      idx++;
    }
  }

  template <std::size_t N>
  [[nodiscard]] const real &get() const noexcept {
    static_assert(N >= 0, "N is not in the valid range");
    static_assert(N < 3, "N is not in the valid range");
    return (*this)(N);
  }

  template <std::size_t N>
  [[nodiscard]] real &get() noexcept {
    static_assert(N >= 0, "N is not in the valid range");
    static_assert(N < 3, "N is not in the valid range");
    return (*this)(N);
  }

  template <std::size_t N>
  [[nodiscard]] real &&get() && {
    static_assert(N >= 0, "N is not in the valid range");
    static_assert(N < 3, "N is not in the valid range");
    return std::move((*this)(N));
  }

  [[nodiscard]] real sum() const noexcept {
    return get<0>() + get<1>() + get<2>();
  }

  [[nodiscard]] triple operator+(const triple &rhs) const noexcept {
    return {get<0>() + rhs.get<0>(), get<1>() + rhs.get<1>(),
            get<2>() + rhs.get<2>()};
  }

  [[nodiscard]] triple operator*(const triple &rhs) const noexcept {
    return {get<0>() * rhs.get<0>(), get<1>() * rhs.get<1>(),
            get<2>() * rhs.get<2>()};
  }
};

namespace std {
template <>
struct tuple_size<triple> : public std::integral_constant<std::size_t, 3> {};

template <std::size_t N>
struct tuple_element<N, triple> {
  using type = real;
};
}  // namespace std

// Store the Jacobian in row major order
class Jacobian : public ND_Array<real, 3, 3> {
 public:
  Jacobian() : ND_Array<real, 3, 3>() {}

  [[nodiscard]] real &get(int r, int c) noexcept {
    return (*this)(r, c);
  }

  [[nodiscard]] const real &get(int row, int col) const noexcept {
    return (*this)(row, col);
  }

  [[nodiscard]] const triple &row(int r) const noexcept {
    return *reinterpret_cast<const triple *>(&get(r, 0));
  }

  [[nodiscard]] triple &row(int r) noexcept {
    return *reinterpret_cast<triple *>(&get(r, 0));
  }

  [[nodiscard]] triple column(int c) const noexcept {
    return triple{get(0, c), get(1, c), get(2, c)};
  }

  [[nodiscard]] triple operator*(const triple &rhs) const noexcept {
    return {(row(0) * rhs).sum(), (row(1) * rhs).sum(), (row(2) * rhs).sum()};
  }
};

#endif  // _MULTIVAR_HPP_
