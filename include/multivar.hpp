
#ifndef _MULTIVAR_HPP_
#define _MULTIVAR_HPP_

#include "constants.hpp"

#include "nd_array/nd_array.hpp"

#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <utility>

#include <cassert>

class triple : public ND_Array<real, 3> {
 public:
  constexpr triple() : ND_Array<real, 3>() {}

  // Make triples compatible with structured bindings and initializer lists
  constexpr triple(std::initializer_list<real> items) : ND_Array<real, 3>() {
    int idx = 0;
    for(real v : items) {
      (*this)(idx) = v;
      idx++;
    }
  }

  constexpr triple &operator=(const real s) noexcept {
    for(int i = 0; i < this->extent(0); i++) {
      (*this)(i) = s;
    }
    return *this;
  }

  template <std::size_t N>
  [[nodiscard]] constexpr const real &get() const noexcept {
    static_assert(N >= 0, "N is not in the valid range");
    static_assert(N < 3, "N is not in the valid range");
    return (*this)(N);
  }

  template <std::size_t N>
  [[nodiscard]] constexpr real &get() noexcept {
    static_assert(N >= 0, "N is not in the valid range");
    static_assert(N < 3, "N is not in the valid range");
    return (*this)(N);
  }

  template <std::size_t N>
  [[nodiscard]] constexpr real &&get() && {
    static_assert(N >= 0, "N is not in the valid range");
    static_assert(N < 3, "N is not in the valid range");
    return std::move((*this)(N));
  }

  [[nodiscard]] constexpr real sum() const noexcept {
    return get<0>() + get<1>() + get<2>();
  }

  constexpr triple &operator+=(const triple &rhs) noexcept {
    for(int i = 0; i < 3; i++) {
      (*this)(i) += rhs(i);
    }
    return *this;
  }

  constexpr triple &operator-=(const triple &rhs) noexcept {
    for(int i = 0; i < 3; i++) {
      (*this)(i) -= rhs(i);
    }
    return *this;
  }

  constexpr triple &operator*=(const triple &rhs) noexcept {
    for(int i = 0; i < 3; i++) {
      (*this)(i) *= rhs(i);
    }
    return *this;
  }

  constexpr triple &operator*=(const real &rhs) noexcept {
    for(int i = 0; i < 3; i++) {
      (*this)(i) *= rhs;
    }
    return *this;
  }

  constexpr triple &operator/=(const triple &rhs) noexcept {
    for(int i = 0; i < 3; i++) {
      (*this)(i) /= rhs(i);
    }
    return *this;
  }

  constexpr triple &operator/=(const real rhs) noexcept {
    const real rhs_inv = 1.0 / rhs;
    for(int i = 0; i < 3; i++) {
      (*this)(i) *= rhs_inv;
    }
    return *this;
  }
};

constexpr triple operator-(const triple &t) noexcept {
  triple s{0.0, 0.0, 0.0};
  s -= t;
  return s;
}

constexpr triple operator+(const triple &lhs, const triple &rhs) noexcept {
  triple s(lhs);
  s += rhs;
  return s;
}

constexpr triple operator-(const triple &lhs, const triple &rhs) noexcept {
  triple s(lhs);
  s -= rhs;
  return s;
}

constexpr triple operator*(const triple &lhs, const triple &rhs) noexcept {
  triple s(lhs);
  s *= rhs;
  return s;
}

constexpr triple operator*(const real lhs, const triple &rhs) noexcept {
  triple s(rhs);
  s *= lhs;
  return s;
}

constexpr triple operator*(const triple &lhs, const real rhs) noexcept {
  triple s(lhs);
  s *= rhs;
  return s;
}

constexpr triple operator/(const triple &lhs, const triple &rhs) noexcept {
  triple s(lhs);
  s /= rhs;
  return s;
}

constexpr triple operator/(const triple &lhs, const real rhs) noexcept {
  triple s(lhs);
  s /= rhs;
  return s;
}

namespace std {
template <>
struct tuple_size<triple> : public std::integral_constant<std::size_t, 3> {};

template <std::size_t N>
struct tuple_element<N, triple> {
  using type = real;
};
}  // namespace std

// Store the Jacobian in row major order
class Jacobian : public ND_Array<triple, 3> {
 public:
  using Base = ND_Array<triple, 3>;

  struct ZeroTag {};
  struct IdentityTag {};

  constexpr Jacobian() : Base() {}

  constexpr Jacobian(const ZeroTag &) : Base() {
    for(int i = 0; i < this->extent(0); i++) {
      for(int j = 0; j < this->extent(1); j++) {
        (*this)(i, j) = 0.0;
      }
    }
  }

  constexpr Jacobian(const IdentityTag &) : Base() {
    for(int i = 0; i < this->extent(0); i++) {
      for(int j = 0; j < this->extent(1); j++) {
        if(i != j) {
          (*this)(i, j) = 0.0;
        } else {
          (*this)(i, j) = 1.0;
        }
      }
    }
  }

  [[nodiscard]] constexpr real &get(int r, int c) noexcept { return row(r)(c); }

  [[nodiscard]] constexpr const real &get(int r, int c) const noexcept {
    return row(r)(c);
  }

  [[nodiscard]] constexpr real &operator()(int r, int c) noexcept {
    return get(r, c);
  }

  [[nodiscard]] constexpr const real &operator()(int r, int c) const noexcept {
    return get(r, c);
  }

  constexpr Jacobian operator+=(const Jacobian &rhs) noexcept {
    for(int i = 0; i < this->extent(0); i++) {
      for(int j = 0; j < this->extent(1); j++) {
        get(i, j) += rhs(i, j);
      }
    }
    return *this;
  }

  constexpr Jacobian operator-=(const Jacobian rhs) noexcept {
    for(int i = 0; i < this->extent(0); i++) {
      for(int j = 0; j < this->extent(1); j++) {
        get(i, j) -= rhs(i, j);
      }
    }
    return *this;
  }

  constexpr Jacobian operator*=(const Jacobian rhs) noexcept {
    for(int r = 0; r < this->extent(0); r++) {
			triple new_row;
      for(int c = 0; c < this->extent(1); c++) {
				new_row(c) = (row(r) * rhs.column(c)).sum();
      }
			for(int c = 0; c < this->extent(1); c++) {
				get(r, c) = new_row(c);
			}
    }
    return *this;
  }

  constexpr Jacobian operator*=(const real rhs) noexcept {
    for(int i = 0; i < this->extent(0); i++) {
      for(int j = 0; j < this->extent(1); j++) {
        get(i, j) *= rhs;
      }
    }
    return *this;
  }

  constexpr Jacobian operator/=(const real rhs) noexcept {
    const real rhs_inv = 1.0 / rhs;
    for(int i = 0; i < this->extent(0); i++) {
      for(int j = 0; j < this->extent(1); j++) {
        get(i, j) *= rhs_inv;
      }
    }
    return *this;
  }

  [[nodiscard]] constexpr const triple &row(int r) const noexcept {
    return (*this)(r);
  }

  [[nodiscard]] constexpr triple &row(int r) noexcept { return (*this)(r); }

  [[nodiscard]] constexpr triple column(int c) const noexcept {
    return triple{get(0, c), get(1, c), get(2, c)};
  }

  [[nodiscard]] constexpr triple operator*(const triple &rhs) const noexcept {
    return {(row(0) * rhs).sum(), (row(1) * rhs).sum(), (row(2) * rhs).sum()};
  }

  [[nodiscard]] constexpr real det() const noexcept {
    real d = 0.0;
    for(int r = 0; r < this->extent(0); r++) {
      d = get(r, 0) * minor(r, 0) - d;
    }
    return d;
  }

  // Use the method of cofactors to compute the inverse
  [[nodiscard]] constexpr Jacobian inverse() const noexcept {
    Jacobian inv;
    assert(det() != 0.0);
    const real d = 1.0 / det();
		real cofactor = 1.0;
    for(int i = 0; i < inv.extent(1); i++) {
			if(i % 2 == 1) {
				cofactor = -1.0;
			}
			else {
				cofactor = 1.0;
			}
      for(int j = 0; j < inv.extent(0); j++) {
        inv(j, i) = cofactor * d * minor(i, j);
				cofactor *= -1.0;
      }
    }
    return inv;
  }

	[[nodiscard]] constexpr Jacobian minors() const noexcept {
		Jacobian M;
		for(int r = 0; r < this->extent(0); r++) {
			for(int c = 0; c < this->extent(1); c++) {
				M(r, c) = minor(r, c);
			}
		}
		return M;
	}

  [[nodiscard]] constexpr real minor(const int r, const int c) const noexcept {
    const auto indices = [](const int omit) {
      if(omit == 0) {
        return std::pair<int, int>(1, 2);
      } else if(omit == 1) {
        return std::pair<int, int>(0, 2);
      } else {
        return std::pair<int, int>(0, 1);
      }
    };
    const auto [left, right] = indices(c);
    const auto [top, bottom] = indices(r);

    return get(top, left) * get(bottom, right) -
           get(top, right) * get(bottom, left);
  }

 protected:
  [[nodiscard]] constexpr triple &operator()(int r) noexcept {
    return Base::operator()(r);
  }

  [[nodiscard]] constexpr const triple &operator()(int r) const noexcept {
    return Base::operator()(r);
  }
};

constexpr Jacobian operator-(const Jacobian &J) noexcept {
  Jacobian K;
  for(int i = 0; i < K.extent(0); i++) {
    for(int j = 0; j < K.extent(1); j++) {
      K(i, j) = -J(i, j);
    }
  }
  return K;
}

constexpr Jacobian operator+(const Jacobian &lhs,
                             const Jacobian &rhs) noexcept {
  Jacobian s(lhs);
  s += rhs;
  return s;
}

constexpr Jacobian operator-(const Jacobian &lhs,
                             const Jacobian &rhs) noexcept {
  Jacobian s(lhs);
  s -= rhs;
  return s;
}

constexpr Jacobian operator*(const Jacobian &lhs,
                             const Jacobian &rhs) noexcept {
  Jacobian s(lhs);
  s *= rhs;
  return s;
}

constexpr Jacobian operator*(const real lhs, const Jacobian &rhs) noexcept {
  Jacobian s(rhs);
  s *= lhs;
  return s;
}

constexpr Jacobian operator*(const Jacobian &lhs, const real rhs) noexcept {
  Jacobian s(lhs);
  s *= rhs;
  return s;
}

#endif  // _MULTIVAR_HPP_
