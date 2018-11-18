
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <type_traits>

namespace ND_Array_internals {

#include "ct_array.hpp"

template <typename A_Type, typename Dims_CT_Array>
class [[nodiscard]] _nd_array {
 public:
  constexpr _nd_array() noexcept {}

  constexpr _nd_array(const _nd_array<A_Type, Dims_CT_Array>
                          &src) noexcept {
    for(int i = 0; i < DIMS::product(); i++) {
      vals[i] = src.vals[i];
    }
  }

  template <typename... int_t>
  [[nodiscard]] constexpr const A_Type &value(
      int_t... indices) const noexcept {
    static_assert(sizeof...(int_t) == DIMS::len(),
                  "Number of indices passed is incorrect");
    return vals[DIMS::slice_idx(indices...)];
  }

  template <typename... int_t>
  [[nodiscard]] constexpr A_Type &value(
      int_t... indices) noexcept {
    static_assert(sizeof...(int_t) == DIMS::len(),
                  "Number of indices passed is incorrect");
    return vals[DIMS::slice_idx(indices...)];
  }

  template <typename... int_t>
  [[nodiscard]] constexpr A_Type &operator()(
      int_t... indices) noexcept {
    return value(indices...);
  }

  template <typename... int_t>
  [[nodiscard]] constexpr const A_Type &operator()(
      int_t... indices) const noexcept {
    return value(indices...);
  }

  template <typename... int_t>
  [[nodiscard]] constexpr _nd_array<
      A_Type, typename forward_truncate_array<
                  sizeof...(int_t), Dims_CT_Array>::type>
      &outer_slice(int_t... indices) const noexcept {
    using truncated_dims =
        typename forward_truncate_array<sizeof...(int_t),
                                        DIMS>::type;
    using ret_type = _nd_array<A_Type, truncated_dims>;
    return *(reinterpret_cast<ret_type *const>(
        &vals[0] + DIMS::slice_idx(indices...)));
  }

  template <typename... int_t>
  [[nodiscard]] _nd_array<
      A_Type, typename forward_truncate_array<
                  sizeof...(int_t), Dims_CT_Array>::type>
      &outer_slice(int_t... indices) noexcept {
    using truncated_dims =
        typename forward_truncate_array<sizeof...(int_t),
                                        DIMS>::type;
    using ret_type = _nd_array<A_Type, truncated_dims>;
    return *(reinterpret_cast<ret_type *>(
        &vals[0] + DIMS::slice_idx(indices...)));
  }

  template <typename Reshaped_Array>
  [[nodiscard]] Reshaped_Array &reshape() noexcept {
    static_assert(
        Reshaped_Array::DIMS::product() == DIMS::product(),
        "Reshaped array is not the same size");
    return *reinterpret_cast<Reshaped_Array *>(this);
  }

  // STL compliance
  [[nodiscard]] constexpr A_Type *data() const noexcept {
    return vals;
  }

  // Provides fast (memory-access wise) iterators
  class iterator {
   public:
    using values_type = A_Type;
    using difference_type = int;
    using reference = A_Type &;
    using pointer = A_Type *;

    constexpr iterator(const iterator &src)
        : _val(src._val), _index(src._index) {}

    constexpr iterator &operator=(const iterator &src) {
      _val = src.val;
      _index = src._index;
      return *this;
    }

    ~iterator() = default;

    constexpr iterator &operator++() {
      _val++;
      _index++;
      return *this;
    }

    [[nodiscard]] constexpr int index(const int dim) const {
      assert(dim < DIMS::len());
      if(dim + 1 == DIMS::len()) {
        return _index % DIMS::value(dim);
      } else {
        return (_index / DIMS::trailing_product(dim + 1)) %
               DIMS::value(dim);
      }
    }

    [[nodiscard]] constexpr reference operator*() const {
      return *_val;
    }

		[[nodiscard]] constexpr bool operator!=(const iterator &cmp) const {
			return _val != cmp._val;
		}

    constexpr void swap(iterator &other) {
      {
        const auto tmp = other._val;
        other._val = _val;
        _val = tmp;
      }
      {
        const auto tmp = other._index;
        other._index = _index;
        _index = tmp;
      }
    }

    constexpr iterator(A_Type *pos)
        : _val(pos), _index(0) {}

   private:
    A_Type *_val;
    int _index;
  };

  [[nodiscard]] constexpr iterator begin() noexcept {
    return iterator(&vals[0]);
  }

  [[nodiscard]] constexpr iterator end() noexcept {
    return iterator(&vals[DIMS::product()]);
  }

  [[nodiscard]] static constexpr int extent(int dim) {
    return DIMS::value(dim);
  }

  [[nodiscard]] static constexpr int dimension() {
    return DIMS::len();
  }

  template <typename _A_Type, typename _Dims_CT_Array>
  friend class _nd_array;

 private:
  using DIMS = Dims_CT_Array;
  A_Type vals[DIMS::product()];
};
}  // namespace ND_Array_internals

template <typename A_Type, int... Dims>
using ND_Array = ND_Array_internals::_nd_array<
    A_Type, ND_Array_internals::CT_Array<int, Dims...> >;

#endif
