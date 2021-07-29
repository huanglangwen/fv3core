#include <algorithm>
#include <array>
#include <cstdint>
#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/block.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/multi_shift.hpp>
#include <gridtools/stencil/common/dim.hpp>
#include <gridtools/stencil/common/extent.hpp>
#include <gridtools/stencil/gpu/launch_kernel.hpp>
#include <gridtools/stencil/gpu/tmp_storage_sid.hpp>

namespace set_initial_vals_impl_ {
using namespace gridtools;
using namespace literals;
using namespace stencil;

using domain_t = std::array<unsigned, 3>;
using i_block_size_t = integral_constant<int, 64>;
using j_block_size_t = integral_constant<int, 8>;

template <class Storage> auto block(Storage storage) {
  return sid::block(std::move(storage),
                    tuple_util::make<hymap::keys<dim::i, dim::j>::values>(
                        i_block_size_t(), j_block_size_t()));
}

namespace tag {
struct delp {};
struct a4_1 {};
struct q {};
struct gam {};
} // namespace tag

template <class Sid> struct loop_139793175270304_f {
  sid::ptr_holder_type<Sid> m_ptr_holder;
  sid::strides_type<Sid> m_strides;
  int k_size;

  template <class Validator>
  GT_FUNCTION_DEVICE void operator()(const int _i_block, const int _j_block,
                                     Validator validator) const {
    auto _ptr = m_ptr_holder();
    sid::shift(_ptr, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides),
               blockIdx.x);
    sid::shift(_ptr, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides),
               blockIdx.y);
    sid::shift(_ptr, sid::get_stride<dim::i>(m_strides), _i_block);
    sid::shift(_ptr, sid::get_stride<dim::j>(m_strides), _j_block);

    const auto gam = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::gam>(
          device::at_key<tag::gam>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };
    const auto delp = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::delp>(
          device::at_key<tag::delp>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };
    const auto q = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::q>(
          device::at_key<tag::q>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };
    const auto a4_1 = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::a4_1>(
          device::at_key<tag::a4_1>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };

    // VerticalLoopSection 139793176513600
    for (int _k_block = 0; _k_block < 1; ++_k_block) {

      // HorizontalExecution 139793177213488
      if (validator(extent<0, 0, 0, 0>())) {
        double grid_ratio0;
        double bet0;
        grid_ratio0 = (delp(0_c, 0_c, 1_c) / delp(0_c, 0_c, 0_c));
        bet0 = (grid_ratio0 * (grid_ratio0 + static_cast<double>(0.5)));
        q(0_c, 0_c, 0_c) = (((((grid_ratio0 + grid_ratio0) *
                               (grid_ratio0 + static_cast<double>(1.0))) *
                              a4_1(0_c, 0_c, 0_c)) +
                             a4_1(0_c, 0_c, 1_c)) /
                            bet0);
        gam(0_c, 0_c, 0_c) =
            ((static_cast<double>(1.0) +
              (grid_ratio0 * (grid_ratio0 + static_cast<double>(1.5)))) /
             bet0);
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), 1_c);
    }

    // VerticalLoopSection 139793176969120
    for (int _k_block = 1; _k_block < k_size + -1; ++_k_block) {

      // HorizontalExecution 139793176138368
      if (validator(extent<0, 0, 0, 0>())) {
        double d5;
        double bet1;
        d5 = (delp(0_c, 0_c, -1_c) / delp(0_c, 0_c, 0_c));
        bet1 = (((static_cast<double>(2.0) + d5) + d5) - gam(0_c, 0_c, -1_c));
        q(0_c, 0_c, 0_c) =
            (((static_cast<double>(3.0) *
               (a4_1(0_c, 0_c, -1_c) + (d5 * a4_1(0_c, 0_c, 0_c)))) -
              q(0_c, 0_c, -1_c)) /
             bet1);
        gam(0_c, 0_c, 0_c) = (d5 / bet1);
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), 1_c);
    }

    // VerticalLoopSection 139793176140192
    for (int _k_block = k_size + -1; _k_block < k_size + 0; ++_k_block) {

      // HorizontalExecution 139793176968400
      if (validator(extent<0, 0, 0, 0>())) {
        double a_bot0;
        double d6;
        d6 = (delp(0_c, 0_c, -2_c) / delp(0_c, 0_c, -1_c));
        a_bot0 =
            (static_cast<double>(1.0) + (d6 * (d6 + static_cast<double>(1.5))));
        q(0_c, 0_c, 0_c) = ((((((static_cast<double>(2.0) * d6) *
                                (d6 + static_cast<double>(1.0))) *
                               a4_1(0_c, 0_c, -1_c)) +
                              a4_1(0_c, 0_c, -2_c)) -
                             (a_bot0 * q(0_c, 0_c, -1_c))) /
                            ((d6 * (d6 + static_cast<double>(0.5))) -
                             (a_bot0 * gam(0_c, 0_c, -1_c))));
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), 1_c);
    }
  }
};

template <class Sid> struct loop_139793175170016_f {
  sid::ptr_holder_type<Sid> m_ptr_holder;
  sid::strides_type<Sid> m_strides;
  int k_size;

  template <class Validator>
  GT_FUNCTION_DEVICE void operator()(const int _i_block, const int _j_block,
                                     Validator validator) const {
    auto _ptr = m_ptr_holder();
    sid::shift(_ptr, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides),
               blockIdx.x);
    sid::shift(_ptr, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides),
               blockIdx.y);
    sid::shift(_ptr, sid::get_stride<dim::i>(m_strides), _i_block);
    sid::shift(_ptr, sid::get_stride<dim::j>(m_strides), _j_block);

    const auto gam = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::gam>(
          device::at_key<tag::gam>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };
    const auto q = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::q>(
          device::at_key<tag::q>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };

    // VerticalLoopSection 139793176512352
    for (int _k_block = k_size + -1 - 1; _k_block >= 0; --_k_block) {

      // HorizontalExecution 139793176852128
      if (validator(extent<0, 0, 0, 0>())) {

        q(0_c, 0_c, 0_c) =
            (q(0_c, 0_c, 0_c) - (gam(0_c, 0_c, 0_c) * q(0_c, 0_c, 1_c)));
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), -1_c);
    }
  }
};

template <class Loop139793175270304, class Loop139793175170016>
struct kernel_139793176804512_f {
  Loop139793175270304 m_139793175270304;
  Loop139793175170016 m_139793175170016;

  template <class Validator>
  GT_FUNCTION_DEVICE void operator()(const int _i_block, const int _j_block,
                                     Validator validator) const {
    m_139793175270304(_i_block, _j_block, validator);
    m_139793175170016(_i_block, _j_block, validator);
  }
};

auto set_initial_vals(domain_t domain) {
  return [domain](auto &&gam, auto &&q, auto &&delp, auto &&a4_1) {
    auto tmp_alloc =
        sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char[]>);
    const int i_size = domain[0];
    const int j_size = domain[1];
    const int k_size = domain[2];
    const int i_blocks = (i_size + i_block_size_t() - 1) / i_block_size_t();
    const int j_blocks = (j_size + j_block_size_t() - 1) / j_block_size_t();

    // kernel 139793176804512

    // vertical loop 139793175270304

    assert((0) >= 0 && (0) < k_size);
    auto offset_139793175270304 =
        tuple_util::make<hymap::keys<dim::k>::values>(0);

    auto composite_139793175270304 =
        sid::composite::make<tag::q, tag::gam, tag::delp, tag::a4_1>(

            block(sid::shift_sid_origin(q, offset_139793175270304)),
            block(sid::shift_sid_origin(gam, offset_139793175270304)),
            block(sid::shift_sid_origin(delp, offset_139793175270304)),
            block(sid::shift_sid_origin(a4_1, offset_139793175270304))

        );
    using composite_139793175270304_t = decltype(composite_139793175270304);
    loop_139793175270304_f<composite_139793175270304_t> loop_139793175270304{
        sid::get_origin(composite_139793175270304),
        sid::get_strides(composite_139793175270304), k_size};

    // vertical loop 139793175170016

    assert((k_size + -1 - 1) >= 0 && (k_size + -1 - 1) < k_size);
    auto offset_139793175170016 =
        tuple_util::make<hymap::keys<dim::k>::values>(k_size + -1 - 1);

    auto composite_139793175170016 = sid::composite::make<tag::q, tag::gam>(

        block(sid::shift_sid_origin(q, offset_139793175170016)),
        block(sid::shift_sid_origin(gam, offset_139793175170016))

    );
    using composite_139793175170016_t = decltype(composite_139793175170016);
    loop_139793175170016_f<composite_139793175170016_t> loop_139793175170016{
        sid::get_origin(composite_139793175170016),
        sid::get_strides(composite_139793175170016), k_size};

    kernel_139793176804512_f<decltype(loop_139793175270304),
                             decltype(loop_139793175170016)>
        kernel_139793176804512{loop_139793175270304, loop_139793175170016};
    gpu_backend::launch_kernel<extent<0, 0, 0, 0>, i_block_size_t::value,
                               j_block_size_t::value>(
        i_size, j_size, 1, kernel_139793176804512, 0);
  };
}
} // namespace set_initial_vals_impl_

using set_initial_vals_impl_::set_initial_vals;
