#include <assert.h>
#include <riscv_vector.h>
#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__rvv_2x4_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    const int32_t* mask_table,
    uint32_t padding_top,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 0);
  assert(padding_top <= 1);
  const size_t nr = __riscv_vsetvlmax_e32m1();
  size_t vl = nr;

  const vuint32m1_t vmask_even = __riscv_vle32_v_u32m1((const uint32_t*) &mask_table[vl - (((input_width & ((vl << 3) - 1)) + vl - 1) >> 3)], vl);
  const vuint32m1_t vmask_odd = __riscv_vle32_v_u32m1((const uint32_t*) &mask_table[vl - ((input_width & ((vl << 3) - 1)) >> 3)], vl);


  const size_t input_decrement = round_down_po2(input_width, vl /* SIMD output width */ * 2 /* subsampling */ * sizeof(float));
  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 5) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i4 = zero;
    }

    vfloat32m1_t vi0x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi1x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi2x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi3x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi4x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

    size_t w = input_width;
    for (; w >= (vl << 3); w -= (vl << 3)) {
      vfloat32m1_t vo0p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
      vfloat32m1_t vo1p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);

      const vfloat32m1x2_t vi0x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i0, vl); i0 += vl << 1;
      const vfloat32m1x2_t vi1x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i1, vl); i1 += vl << 1;
      const vfloat32m1x2_t vi2x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i2, vl); i2 += vl << 1;
      const vfloat32m1x2_t vi3x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i3, vl); i3 += vl << 1;
      const vfloat32m1x2_t vi4x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i4, vl); i4 += vl << 1;

      vfloat32m1_t vo0p1 = __riscv_vfmul_vf_f32m1(__riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 0), weights[2], vl);
      vfloat32m1_t vo1p1 = __riscv_vfmul_vf_f32m1(__riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 0), weights[2], vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[5], __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 0), vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[5], __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 0), vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[8], __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 0), vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[8], __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 0), vl);

      const vfloat32m1_t vi0x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x1357, vl-1, vl),
                                                            __riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 1), 1, vl);
      vi0x1357 = __riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 1);
      const vfloat32m1_t vi1x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x1357, vl-1, vl),
                                                            __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 1), 1, vl);
      vi1x1357 = __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 1);
      const vfloat32m1_t vi2x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x1357, vl-1, vl),
                                                            __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1), 1, vl);
      vi2x1357 = __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1);
      const vfloat32m1_t vi3x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x1357, vl-1, vl),
                                                            __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 1), 1, vl);
      vi3x1357 = __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 1);
      const vfloat32m1_t vi4x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x1357, vl-1, vl),
                                                            __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 1), 1, vl);
      vi4x1357 = __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 1);

      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[1], vi0x79BD, vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[1], vi2x79BD, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[4], vi1x79BD, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[4], vi3x79BD, vl);

      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[7], vi2x79BD, vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[7], vi4x79BD, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[3], __riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 1), vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[3], __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1), vl);

      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[6], __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 1), vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[6], __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 1), vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[9], __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1), vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[9], __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 1), vl);

      vo0p0 = __riscv_vfadd_vv_f32m1(vo0p0, vo0p1, vl);
      vo1p0 = __riscv_vfadd_vv_f32m1(vo1p0, vo1p1, vl);

      vfloat32m1_t vo0 = __riscv_vfmax_vf_f32m1(vo0p0, params->scalar.min, vl);
      vfloat32m1_t vo1 = __riscv_vfmax_vf_f32m1(vo1p0, params->scalar.min, vl);

      vo0 = __riscv_vfmin_vf_f32m1(vo0, params->scalar.max, vl);
      vo1 = __riscv_vfmin_vf_f32m1(vo1, params->scalar.max, vl);

      __riscv_vse32_v_f32m1(o1, vo1, vl); o1 += vl;
      __riscv_vse32_v_f32m1(o0, vo0, vl); o0 += vl;
    }
    // Last block has 0-7 pixels to process.
    assert(w < (vl << 3));
    if XNN_LIKELY(w != 0) {
      vfloat32m1_t vo0p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
      vfloat32m1_t vo1p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);

      const vfloat32m1x2_t vi0x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i0, vl);
      const vfloat32m1x2_t vi1x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i1, vl);
      const vfloat32m1x2_t vi2x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i2, vl);
      const vfloat32m1x2_t vi3x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i3, vl);
      const vfloat32m1x2_t vi4x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i4, vl);

      const vfloat32m1_t vi0x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 0)), vl));
      const vfloat32m1_t vi0x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd,  __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 1)), vl));
      const vfloat32m1_t vi1x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 0)), vl));
      const vfloat32m1_t vi1x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd,  __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 1)), vl));
      const vfloat32m1_t vi2x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 0)), vl));
      const vfloat32m1_t vi2x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd,  __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1)), vl));
      const vfloat32m1_t vi3x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 0)), vl));
      const vfloat32m1_t vi3x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd,  __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 1)), vl));
      const vfloat32m1_t vi4x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 0)), vl));
      const vfloat32m1_t vi4x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd,  __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 1)), vl));

      vfloat32m1_t vo0p1 = __riscv_vfmul_vf_f32m1(vi0x8ACE, weights[2], vl);
      vfloat32m1_t vo1p1 = __riscv_vfmul_vf_f32m1(vi2x8ACE, weights[2], vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[5], vi1x8ACE, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[5], vi3x8ACE, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[8], vi2x8ACE, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[8], vi4x8ACE, vl);

      const vfloat32m1_t vi0x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x1357, vl-1, vl), vi0x9BDF, 1, vl);
      const vfloat32m1_t vi1x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x1357, vl-1, vl), vi1x9BDF, 1, vl);
      const vfloat32m1_t vi2x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x1357, vl-1, vl), vi2x9BDF, 1, vl);
      const vfloat32m1_t vi3x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x1357, vl-1, vl), vi3x9BDF, 1, vl);
      const vfloat32m1_t vi4x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x1357, vl-1, vl), vi4x9BDF, 1, vl);

      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[1], vi0x79BD, vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[1], vi2x79BD, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[4], vi1x79BD, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[4], vi3x79BD, vl);

      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[7], vi2x79BD, vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[7], vi4x79BD, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[3], vi0x9BDF, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[3], vi2x9BDF, vl);

      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[6], vi1x9BDF, vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[6], vi3x9BDF, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[9], vi2x9BDF, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[9], vi4x9BDF, vl);

      vo0p0 = __riscv_vfadd_vv_f32m1(vo0p0, vo0p1, vl);
      vo1p0 = __riscv_vfadd_vv_f32m1(vo1p0, vo1p1, vl);

      vfloat32m1_t vo0 = __riscv_vfmax_vf_f32m1(vo0p0, params->scalar.min, vl);
      vfloat32m1_t vo1 = __riscv_vfmax_vf_f32m1(vo1p0, params->scalar.min, vl);

      vo0 = __riscv_vfmin_vf_f32m1(vo0, params->scalar.max, vl);
      vo1 = __riscv_vfmin_vf_f32m1(vo1, params->scalar.max, vl);

      w += 1 * sizeof(float);
      __riscv_vse32_v_f32m1(o1, vo1, w >> 3); o1 += w >> 3;
      __riscv_vse32_v_f32m1(o0, vo0, w >> 3); o0 += w >> 3;

    }
    i0 = (const float*) ((uintptr_t) i4 - input_decrement);
    i1 = (const float*) ((uintptr_t) i0 + input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + output_width);

    output_height = doz(output_height, 2);
    padded_input_height = doz(padded_input_height, 4);
  } while (output_height != 0);
}
