#include <assert.h>

#include <riscv_vector.h>
#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"

void xnn_f32_dwconv2d_chw_ukernel_3x3p1__rvv_3x4_acc2(
  size_t input_height,
  size_t input_width,
  const float* input,
  const float* weights,
  const float* zero,
  float* output,
  const int32_t* mask_table,
  uint32_t padding_top,
  const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
assert(input_height != 0);
assert(input_width != 0);
assert(input_width % sizeof(float) == 0);
assert(padding_top == 1);

const size_t nr = __riscv_vsetvlmax_e32m1();
const size_t input_decrement = round_up_po2(input_width, nr * sizeof(float));

const float* i0 = zero;
const float* i1 = input;
const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

float* o0 = output;
float* o1 = (float*) ((uintptr_t) o0 + input_width);
float* o2 = (float*) ((uintptr_t) o1 + input_width);

size_t output_height = input_height;
do {
  if XNN_UNPREDICTABLE(output_height < 2) {
    i2 = zero;
    o1 = o0;
  }
  if XNN_UNPREDICTABLE(output_height < 3) {
    i3 = zero;
    o2 = o1;
  }
  if XNN_UNPREDICTABLE(output_height < 4) {
    i4 = zero;
  }
  size_t w = input_width >> 2;
  size_t vl = nr;
  vfloat32m1_t vi0x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi1x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi2x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi3x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi4x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

  vfloat32m1_t vi0x4567 = __riscv_vle32_v_f32m1(i0, vl); i0 += vl;
  vfloat32m1_t vi1x4567 = __riscv_vle32_v_f32m1(i1, vl); i1 += vl;
  vfloat32m1_t vi2x4567 = __riscv_vle32_v_f32m1(i2, vl); i2 += vl;
  vfloat32m1_t vi3x4567 = __riscv_vle32_v_f32m1(i3, vl); i3 += vl;
  vfloat32m1_t vi4x4567 = __riscv_vle32_v_f32m1(i4, vl); i4 += vl;


  for (; w > nr; w -= nr) {
      vfloat32m1_t vo0p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
      vfloat32m1_t vo1p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
      vfloat32m1_t vo2p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);

      vfloat32m1_t vi0x89AB = __riscv_vle32_v_f32m1(i0, vl); i0 += vl;
      vfloat32m1_t vi1x89AB = __riscv_vle32_v_f32m1(i1, vl); i1 += vl;
      vfloat32m1_t vi2x89AB = __riscv_vle32_v_f32m1(i2, vl); i2 += vl;
      vfloat32m1_t vi3x89AB = __riscv_vle32_v_f32m1(i3, vl); i3 += vl;
      vfloat32m1_t vi4x89AB = __riscv_vle32_v_f32m1(i4, vl); i4 += vl;

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[2], vi0x4567, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[2], vi1x4567, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[2], vi2x4567, vl);

      vfloat32m1_t vo0p1 = __riscv_vfmul_vf_f32m1(vi1x4567, weights[5], vl);
      vfloat32m1_t vo1p1 = __riscv_vfmul_vf_f32m1(vi2x4567, weights[5], vl);
      vfloat32m1_t vo2p1 = __riscv_vfmul_vf_f32m1(vi3x4567, weights[5], vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[8], vi2x4567, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[8], vi3x4567, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[8], vi4x4567, vl);


      const vfloat32m1_t vi0x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x0123, vl-1, vl), vi0x4567, 1, vl);
      const vfloat32m1_t vi1x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x0123, vl-1, vl), vi1x4567, 1, vl);
      const vfloat32m1_t vi2x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x0123, vl-1, vl), vi2x4567, 1, vl);
      const vfloat32m1_t vi3x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x0123, vl-1, vl), vi3x4567, 1, vl);
      const vfloat32m1_t vi4x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x0123, vl-1, vl), vi4x4567, 1, vl);


      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[1], vi0x3456, vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[1], vi1x3456, vl);
      vo2p1 = __riscv_vfmacc_vf_f32m1(vo2p1, weights[1], vi2x3456, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[4], vi1x3456, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[4], vi2x3456, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[4], vi3x3456, vl);

      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[7], vi2x3456, vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[7], vi3x3456, vl);
      vo2p1 = __riscv_vfmacc_vf_f32m1(vo2p1, weights[7], vi4x3456, vl);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;

      const vfloat32m1_t vi0x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x4567, 1, vl), vi0x89AB, vl-1, vl);
      const vfloat32m1_t vi1x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x4567, 1, vl), vi1x89AB, vl-1, vl);
      const vfloat32m1_t vi2x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x4567, 1, vl), vi2x89AB, vl-1, vl);
      const vfloat32m1_t vi3x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x4567, 1, vl), vi3x89AB, vl-1, vl);
      const vfloat32m1_t vi4x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x4567, 1, vl), vi4x89AB, vl-1, vl);


      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[3], vi0x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[3], vi1x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[3], vi2x5678, vl);

      vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[6], vi1x5678, vl);
      vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[6], vi2x5678, vl);
      vo2p1 = __riscv_vfmacc_vf_f32m1(vo2p1, weights[6], vi3x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[9], vi2x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[9], vi3x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[9], vi4x5678, vl);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;

      vo0p0 = __riscv_vfadd_vv_f32m1(vo0p0, vo0p1, vl);
      vo1p0 = __riscv_vfadd_vv_f32m1(vo1p0, vo1p1, vl);
      vo2p0 = __riscv_vfadd_vv_f32m1(vo2p0, vo2p1, vl);

      vfloat32m1_t vo0 = __riscv_vfmax_vf_f32m1(vo0p0, params->scalar.min, vl);
      vfloat32m1_t vo1 = __riscv_vfmax_vf_f32m1(vo1p0, params->scalar.min, vl);
      vfloat32m1_t vo2 = __riscv_vfmax_vf_f32m1(vo2p0, params->scalar.min, vl);

      vo0 = __riscv_vfmin_vf_f32m1(vo0, params->scalar.max, vl);
      vo1 = __riscv_vfmin_vf_f32m1(vo1, params->scalar.max, vl);
      vo2 = __riscv_vfmin_vf_f32m1(vo2, params->scalar.max, vl);

      __riscv_vse32_v_f32m1(o2, vo2, vl); o2 += vl;
      __riscv_vse32_v_f32m1(o1, vo1, vl); o1 += vl;
      __riscv_vse32_v_f32m1(o0, vo0, vl); o0 += vl;
  }
  // Always process the last block of 1..4 pixels.
  assert(w >= 1);
  assert(w <= nr);
  {
    vfloat32m1_t vo0p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
    vfloat32m1_t vo1p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
    vfloat32m1_t vo2p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
    const vuint32m1_t vmask = __riscv_vle32_v_u32m1((const uint32_t*) &mask_table[(nr-1) - (((input_width >> 2) - 1) & (nr-1))], vl);
    vi0x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi0x4567), vl));
    vi1x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi1x4567), vl));
    vi2x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi2x4567), vl));
    vi3x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi3x4567), vl));
    vi4x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi4x4567), vl));

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[2], vi0x4567, vl);
    vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[2], vi1x4567, vl);
    vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[2], vi2x4567, vl);

    vfloat32m1_t vo0p1 = __riscv_vfmul_vf_f32m1(vi1x4567, weights[5], vl);
    vfloat32m1_t vo1p1 = __riscv_vfmul_vf_f32m1(vi2x4567, weights[5], vl);
    vfloat32m1_t vo2p1 = __riscv_vfmul_vf_f32m1(vi3x4567, weights[5], vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[8], vi2x4567, vl);
    vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[8], vi3x4567, vl);
    vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[8], vi4x4567, vl);

    const vfloat32m1_t vi0x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x0123, vl-1, vl), vi0x4567, 1, vl);
    const vfloat32m1_t vi1x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x0123, vl-1, vl), vi1x4567, 1, vl);
    const vfloat32m1_t vi2x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x0123, vl-1, vl), vi2x4567, 1, vl);
    const vfloat32m1_t vi3x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x0123, vl-1, vl), vi3x4567, 1, vl);
    const vfloat32m1_t vi4x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x0123, vl-1, vl), vi4x4567, 1, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[1], vi0x3456, vl);
    vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[1], vi1x3456, vl);
    vo2p1 = __riscv_vfmacc_vf_f32m1(vo2p1, weights[1], vi2x3456, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[4], vi1x3456, vl);
    vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[4], vi2x3456, vl);
    vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[4], vi3x3456, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[7], vi2x3456, vl);
    vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[7], vi3x3456, vl);
    vo2p1 = __riscv_vfmacc_vf_f32m1(vo2p1, weights[7], vi4x3456, vl);

    const vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);

    const vfloat32m1_t vi0x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x4567, 1, vl), vzero, vl-1, vl);
    const vfloat32m1_t vi1x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x4567, 1, vl), vzero, vl-1, vl);
    const vfloat32m1_t vi2x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x4567, 1, vl), vzero, vl-1, vl);
    const vfloat32m1_t vi3x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x4567, 1, vl), vzero, vl-1, vl);
    const vfloat32m1_t vi4x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x4567, 1, vl), vzero, vl-1, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[3], vi0x5678, vl);
    vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[3], vi1x5678, vl);
    vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[3], vi2x5678, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[6], vi1x5678, vl);
    vo1p1 = __riscv_vfmacc_vf_f32m1(vo1p1, weights[6], vi2x5678, vl);
    vo2p1 = __riscv_vfmacc_vf_f32m1(vo2p1, weights[6], vi3x5678, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[9], vi2x5678, vl);
    vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[9], vi3x5678, vl);
    vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[9], vi4x5678, vl);

    vo0p0 = __riscv_vfadd_vv_f32m1(vo0p0, vo0p1, vl);
    vo1p0 = __riscv_vfadd_vv_f32m1(vo1p0, vo1p1, vl);
    vo2p0 = __riscv_vfadd_vv_f32m1(vo2p0, vo2p1, vl);

    vfloat32m1_t vo0 = __riscv_vfmax_vf_f32m1(vo0p0, params->scalar.min, vl);
    vfloat32m1_t vo1 = __riscv_vfmax_vf_f32m1(vo1p0, params->scalar.min, vl);
    vfloat32m1_t vo2 = __riscv_vfmax_vf_f32m1(vo2p0, params->scalar.min, vl);

    vo0 = __riscv_vfmin_vf_f32m1(vo0, params->scalar.max, vl);
    vo1 = __riscv_vfmin_vf_f32m1(vo1, params->scalar.max, vl);
    vo2 = __riscv_vfmin_vf_f32m1(vo2, params->scalar.max, vl);

    __riscv_vse32_v_f32m1(o2, vo2, w); o2 += w;
    __riscv_vse32_v_f32m1(o1, vo1, w); o1 += w;
    __riscv_vse32_v_f32m1(o0, vo0, w); o0 += w;
  }

  i0 = (const float*) ((uintptr_t) i3 - input_decrement);
  i1 = (const float*) ((uintptr_t) i4 - input_decrement);
  i2 = (const float*) ((uintptr_t) i1 + input_width);
  i3 = (const float*) ((uintptr_t) i2 + input_width);
  i4 = (const float*) ((uintptr_t) i3 + input_width);

  o0 = o2;
  o1 = (float*) ((uintptr_t) o0 + input_width);
  o2 = (float*) ((uintptr_t) o1 + input_width);

  output_height = doz(output_height, 3);
} while (output_height != 0);
}
