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
struct gam {};
struct delp {};
struct a4_1 {};
struct q {};
} // namespace tag

template <class Sid> struct loop_140222093624368_f {
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

    const auto delp = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::delp>(
          device::at_key<tag::delp>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };
    const auto a4_1 = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::a4_1>(
          device::at_key<tag::a4_1>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };
    const auto q = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::q>(
          device::at_key<tag::q>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };
    const auto gam = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::gam>(
          device::at_key<tag::gam>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };

    double gam1m1;
    double gam1p0;

    double q2m1;
    double q2p0;

    double delp1m2;
    double delp1m1;
    double delp1p0;
    double delp1p1;

    double a4_3m2;
    double a4_3m1;
    double a4_3p0;
    double a4_3p1;

    // VerticalLoopSection 140222093527648
    for (int _k_block = 0; _k_block < 1; ++_k_block) {

      // HorizontalExecution 140222092635632
      if (validator(extent<0, 0, 0, 0>())) {
        double grid_ratio0;
        double bet0;
        a4_3p0 = a4_1(0_c, 0_c, 0_c);
        a4_3p1 = a4_1(0_c, 0_c, 1_c);
        delp1p0 = delp(0_c, 0_c, 0_c);
        delp1p1 = delp(0_c, 0_c, 1_c);
        grid_ratio0 = (delp1p1 / delp1p0);
        bet0 = (grid_ratio0 * (grid_ratio0 + static_cast<double>(0.5)));
        q2p0 = (((((grid_ratio0 + grid_ratio0) *
                   (grid_ratio0 + static_cast<double>(1.0))) *
                  a4_3p0) +
                 a4_3p1) /
                bet0);
        gam1p0 = ((static_cast<double>(1.0) +
                   (grid_ratio0 * (grid_ratio0 + static_cast<double>(1.5)))) /
                  bet0);
        gam(0_c, 0_c, 0_c) = gam1p0;
        q(0_c, 0_c, 0_c) = q2p0;
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), 1_c);

      gam1m1 = gam1p0;

      q2m1 = q2p0;

      delp1m2 = delp1m1;
      delp1m1 = delp1p0;
      delp1p0 = delp1p1;

      a4_3m2 = a4_3m1;
      a4_3m1 = a4_3p0;
      a4_3p0 = a4_3p1;
    }

    // VerticalLoopSection 140221712878368
    for (int _k_block = 1; _k_block < k_size + -1; ++_k_block) {

      // HorizontalExecution 140222093794368
      if (validator(extent<0, 0, 0, 0>())) {
        double d5;
        double bet1;
        d5 = (delp1m1 / delp1p0);
        bet1 = (((static_cast<double>(2.0) + d5) + d5) - gam1m1);
        q2p0 = (((static_cast<double>(3.0) * (a4_3m1 + (d5 * a4_3p0))) - q2m1) /
                bet1);
        gam1p0 = (d5 / bet1);
        gam(0_c, 0_c, 0_c) = gam1p0;
        q(0_c, 0_c, 0_c) = q2p0;
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), 1_c);

      gam1m1 = gam1p0;

      q2m1 = q2p0;

      delp1m2 = delp1m1;
      delp1m1 = delp1p0;
      delp1p0 = delp1p1;

      a4_3m2 = a4_3m1;
      a4_3m1 = a4_3p0;
      a4_3p0 = a4_3p1;
    }

    // VerticalLoopSection 140221712877168
    for (int _k_block = k_size + -1; _k_block < k_size + 0; ++_k_block) {

      // HorizontalExecution 140221712939232
      if (validator(extent<0, 0, 0, 0>())) {
        double a_bot0;
        double d6;
        d6 = (delp1m2 / delp1m1);
        a_bot0 =
            (static_cast<double>(1.0) + (d6 * (d6 + static_cast<double>(1.5))));
        q2p0 = ((((((static_cast<double>(2.0) * d6) *
                    (d6 + static_cast<double>(1.0))) *
                   a4_3m1) +
                  a4_3m2) -
                 (a_bot0 * q2m1)) /
                ((d6 * (d6 + static_cast<double>(0.5))) - (a_bot0 * gam1m1)));
        q(0_c, 0_c, 0_c) = q2p0;
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), 1_c);

      gam1m1 = gam1p0;

      q2m1 = q2p0;

      delp1m2 = delp1m1;
      delp1m1 = delp1p0;
      delp1p0 = delp1p1;

      a4_3m2 = a4_3m1;
      a4_3m1 = a4_3p0;
      a4_3p0 = a4_3p1;
    }
  }
};

template <class Sid> struct loop_140222092635296_f {
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

    const auto q = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::q>(
          device::at_key<tag::q>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };
    const auto gam = [&](auto i, auto j, auto k) -> auto && {
      return *sid::multi_shifted<tag::gam>(
          device::at_key<tag::gam>(_ptr), m_strides,
          tuple_util::device::make<
              hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k));
    };

    double q3p0;
    double q3p1;

    // VerticalLoopSection 140222093794896
    for (int _k_block = k_size + -1 - 1; _k_block >= k_size + -2; --_k_block) {

      // HorizontalExecution 140222093794080
      if (validator(extent<0, 0, 0, 0>())) {

        q3p1 = q(0_c, 0_c, 1_c);
        q3p0 = q(0_c, 0_c, 0_c);
        q3p0 = (q3p0 - (gam(0_c, 0_c, 0_c) * q3p1));
        q(0_c, 0_c, 0_c) = q3p0;
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), -1_c);

      q3p1 = q3p0;
    }

    // VerticalLoopSection 140221712637712
    for (int _k_block = k_size + -2 - 1; _k_block >= 0; --_k_block) {

      // HorizontalExecution 140221712637856
      if (validator(extent<0, 0, 0, 0>())) {

        q3p0 = q(0_c, 0_c, 0_c);
        q3p0 = (q3p0 - (gam(0_c, 0_c, 0_c) * q3p1));
        q(0_c, 0_c, 0_c) = q3p0;
      }

      sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), -1_c);

      q3p1 = q3p0;
    }
  }
};

template <class Loop140222093624368, class Loop140222092635296>
struct kernel_140222093623504_f {
  Loop140222093624368 m_140222093624368;
  Loop140222092635296 m_140222092635296;

  template <class Validator>
  GT_FUNCTION_DEVICE void operator()(const int _i_block, const int _j_block,
                                     Validator validator) const {
    m_140222093624368(_i_block, _j_block, validator);
    m_140222092635296(_i_block, _j_block, validator);
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

    // kernel 140222093623504

    // vertical loop 140222093624368

    assert((0) >= 0 && (0) < k_size);
    auto offset_140222093624368 =
        tuple_util::make<hymap::keys<dim::k>::values>(0);

    auto composite_140222093624368 =
        sid::composite::make<tag::gam, tag::delp, tag::a4_1, tag::q>(

            block(sid::shift_sid_origin(gam, offset_140222093624368)),
            block(sid::shift_sid_origin(delp, offset_140222093624368)),
            block(sid::shift_sid_origin(a4_1, offset_140222093624368)),
            block(sid::shift_sid_origin(q, offset_140222093624368))

        );
    using composite_140222093624368_t = decltype(composite_140222093624368);
    loop_140222093624368_f<composite_140222093624368_t> loop_140222093624368{
        sid::get_origin(composite_140222093624368),
        sid::get_strides(composite_140222093624368), k_size};

    // vertical loop 140222092635296

    assert((k_size + -1 - 1) >= 0 && (k_size + -1 - 1) < k_size);
    auto offset_140222092635296 =
        tuple_util::make<hymap::keys<dim::k>::values>(k_size + -1 - 1);

    auto composite_140222092635296 = sid::composite::make<tag::gam, tag::q>(

        block(sid::shift_sid_origin(gam, offset_140222092635296)),
        block(sid::shift_sid_origin(q, offset_140222092635296))

    );
    using composite_140222092635296_t = decltype(composite_140222092635296);
    loop_140222092635296_f<composite_140222092635296_t> loop_140222092635296{
        sid::get_origin(composite_140222092635296),
        sid::get_strides(composite_140222092635296), k_size};

    kernel_140222093623504_f<decltype(loop_140222093624368),
                             decltype(loop_140222092635296)>
        kernel_140222093623504{loop_140222093624368, loop_140222092635296};
    gpu_backend::launch_kernel<extent<0, 0, 0, 0>, i_block_size_t::value,
                               j_block_size_t::value>(
        i_size, j_size, 1, kernel_140222093623504, 0);
  };
}
} // namespace set_initial_vals_impl_

using set_initial_vals_impl_::set_initial_vals;
