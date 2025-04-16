#include <assert.h>
#include <riscv_vector.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_5x5p2__rvv_4x4(
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
  assert(padding_top == 2);
  const size_t nr = __riscv_vsetvlmax_e32m1();
  size_t vl = nr;

  const size_t input_decrement = round_up_po2(input_width, vl * sizeof(float));

  const float* i0 = zero;
  const float* i1 = zero;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i5 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i6 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
      i7 = zero;
    }

    vfloat32m1_t vi0x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi1x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi2x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi3x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi4x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi5x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi6x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vi7x0123 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t vi0x4567 = __riscv_vle32_v_f32m1(i0, vl); i0 += vl;
    vfloat32m1_t vi1x4567 = __riscv_vle32_v_f32m1(i1, vl); i1 += vl;
    vfloat32m1_t vi2x4567 = __riscv_vle32_v_f32m1(i2, vl); i2 += vl;
    vfloat32m1_t vi3x4567 = __riscv_vle32_v_f32m1(i3, vl); i3 += vl;
    vfloat32m1_t vi4x4567 = __riscv_vle32_v_f32m1(i4, vl); i4 += vl;
    vfloat32m1_t vi5x4567 = __riscv_vle32_v_f32m1(i5, vl); i5 += vl;
    vfloat32m1_t vi6x4567 = __riscv_vle32_v_f32m1(i6, vl); i6 += vl;
    vfloat32m1_t vi7x4567 = __riscv_vle32_v_f32m1(i7, vl); i7 += vl;

    size_t w = input_width;
    for (; w > vl << 2; w -= vl << 2) {
        vfloat32m1_t vo0p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
        vfloat32m1_t vo1p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
        vfloat32m1_t vo2p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
        vfloat32m1_t vo3p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);

        vfloat32m1_t vi0x89AB = __riscv_vle32_v_f32m1(i0, vl); i0 += vl;
        vfloat32m1_t vi1x89AB = __riscv_vle32_v_f32m1(i1, vl); i1 += vl;
        vfloat32m1_t vi2x89AB = __riscv_vle32_v_f32m1(i2, vl); i2 += vl;
        vfloat32m1_t vi3x89AB = __riscv_vle32_v_f32m1(i3, vl); i3 += vl;
        vfloat32m1_t vi4x89AB = __riscv_vle32_v_f32m1(i4, vl); i4 += vl;
        vfloat32m1_t vi5x89AB = __riscv_vle32_v_f32m1(i5, vl); i5 += vl;
        vfloat32m1_t vi6x89AB = __riscv_vle32_v_f32m1(i6, vl); i6 += vl;
        vfloat32m1_t vi7x89AB = __riscv_vle32_v_f32m1(i7, vl); i7 += vl;

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[3], vi0x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[3], vi1x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[3], vi2x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[3], vi3x4567, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[8], vi1x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[8], vi2x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[8], vi3x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[8], vi4x4567, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[13], vi2x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[13], vi3x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[13], vi4x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[13], vi5x4567, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[18], vi3x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[18], vi4x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[18], vi5x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[18], vi6x4567, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[23], vi4x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[23], vi5x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[23], vi6x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[23], vi7x4567, vl);

        const vfloat32m1_t vi0x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x0123, vl-1, vl), vi0x4567, 1, vl);
        const vfloat32m1_t vi1x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x0123, vl-1, vl), vi1x4567, 1, vl);
        const vfloat32m1_t vi2x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x0123, vl-1, vl), vi2x4567, 1, vl);
        const vfloat32m1_t vi3x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x0123, vl-1, vl), vi3x4567, 1, vl);
        const vfloat32m1_t vi4x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x0123, vl-1, vl), vi4x4567, 1, vl);
        const vfloat32m1_t vi5x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi5x0123, vl-1, vl), vi5x4567, 1, vl);
        const vfloat32m1_t vi6x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi6x0123, vl-1, vl), vi6x4567, 1, vl);
        const vfloat32m1_t vi7x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi7x0123, vl-1, vl), vi7x4567, 1, vl);


      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[2], vi0x3456, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[2], vi1x3456, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[2], vi2x3456, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[2], vi3x3456, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[7], vi1x3456, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[7], vi2x3456, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[7], vi3x3456, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[7], vi4x3456, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[12], vi2x3456, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[12], vi3x3456, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[12], vi4x3456, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[12], vi5x3456, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[17], vi3x3456, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[17], vi4x3456, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[17], vi5x3456, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[17], vi6x3456, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[22], vi4x3456, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[22], vi5x3456, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[22], vi6x3456, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[22], vi7x3456, vl);

      const vfloat32m1_t vi0x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x0123, vl-2, vl), vi0x4567, 2, vl);
      vi0x0123 = vi0x4567;
      const vfloat32m1_t vi1x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x0123, vl-2, vl), vi1x4567, 2, vl);
      vi1x0123 = vi1x4567;
      const vfloat32m1_t vi2x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x0123, vl-2, vl), vi2x4567, 2, vl);
      vi2x0123 = vi2x4567;
      const vfloat32m1_t vi3x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x0123, vl-2, vl), vi3x4567, 2, vl);
      vi3x0123 = vi3x4567;
      const vfloat32m1_t vi4x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x0123, vl-2, vl), vi4x4567, 2, vl);
      vi4x0123 = vi4x4567;
      const vfloat32m1_t vi5x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi5x0123, vl-2, vl), vi5x4567, 2, vl);
      vi5x0123 = vi5x4567;
      const vfloat32m1_t vi6x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi6x0123, vl-2, vl), vi6x4567, 2, vl);
      vi6x0123 = vi6x4567;
      const vfloat32m1_t vi7x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi7x0123, vl-2, vl), vi7x4567, 2, vl);
      vi7x0123 = vi7x4567;

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[1], vi0x2345, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[1], vi1x2345, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[1], vi2x2345, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[1], vi3x2345, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[6], vi1x2345, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[6], vi2x2345, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[6], vi3x2345, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[6], vi4x2345, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[11], vi2x2345, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[11], vi3x2345, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[11], vi4x2345, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[11], vi5x2345, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[16], vi3x2345, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[16], vi4x2345, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[16], vi5x2345, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[16], vi6x2345, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[21], vi4x2345, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[21], vi5x2345, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[21], vi6x2345, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[21], vi7x2345, vl);

      const vfloat32m1_t vi0x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x4567, 1, vl), vi0x89AB, vl-1, vl);
      const vfloat32m1_t vi1x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x4567, 1, vl), vi1x89AB, vl-1, vl);
      const vfloat32m1_t vi2x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x4567, 1, vl), vi2x89AB, vl-1, vl);
      const vfloat32m1_t vi3x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x4567, 1, vl), vi3x89AB, vl-1, vl);
      const vfloat32m1_t vi4x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x4567, 1, vl), vi4x89AB, vl-1, vl);
      const vfloat32m1_t vi5x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi5x4567, 1, vl), vi5x89AB, vl-1, vl);
      const vfloat32m1_t vi6x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi6x4567, 1, vl), vi6x89AB, vl-1, vl);
      const vfloat32m1_t vi7x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi7x4567, 1, vl), vi7x89AB, vl-1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[4], vi0x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[4], vi1x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[4], vi2x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[4], vi3x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[9], vi1x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[9], vi2x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[9], vi3x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[9], vi4x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[14], vi2x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[14], vi3x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[14], vi4x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[14], vi5x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[19], vi3x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[19], vi4x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[19], vi5x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[19], vi6x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[24], vi4x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[24], vi5x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[24], vi6x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[24], vi7x5678, vl);

      const vfloat32m1_t vi0x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x4567, 2, vl), vi0x89AB, vl-2, vl);
      vi0x4567 = vi0x89AB;
      const vfloat32m1_t vi1x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x4567, 2, vl), vi1x89AB, vl-2, vl);
      vi1x4567 = vi1x89AB;
      const vfloat32m1_t vi2x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x4567, 2, vl), vi2x89AB, vl-2, vl);
      vi2x4567 = vi2x89AB;
      const vfloat32m1_t vi3x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x4567, 2, vl), vi3x89AB, vl-2, vl);
      vi3x4567 = vi3x89AB;
      const vfloat32m1_t vi4x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x4567, 2, vl), vi4x89AB, vl-2, vl);
      vi4x4567 = vi4x89AB;
      const vfloat32m1_t vi5x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi5x4567, 2, vl), vi5x89AB, vl-2, vl);
      vi5x4567 = vi5x89AB;
      const vfloat32m1_t vi6x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi6x4567, 2, vl), vi6x89AB, vl-2, vl);
      vi6x4567 = vi6x89AB;
      const vfloat32m1_t vi7x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi7x4567, 2, vl), vi7x89AB, vl-2, vl);
      vi7x4567 = vi7x89AB;

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[5], vi0x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[5], vi1x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[5], vi2x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[5], vi3x6789, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[10], vi1x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[10], vi2x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[10], vi3x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[10], vi4x6789, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[15], vi2x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[15], vi3x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[15], vi4x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[15], vi5x6789, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[20], vi3x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[20], vi4x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[20], vi5x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[20], vi6x6789, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[25], vi4x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[25], vi5x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[25], vi6x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[25], vi7x6789, vl);


      vfloat32m1_t vo0 = __riscv_vfmax_vf_f32m1(vo0p0, params->scalar.min, vl);
      vfloat32m1_t vo1 = __riscv_vfmax_vf_f32m1(vo1p0, params->scalar.min, vl);
      vfloat32m1_t vo2 = __riscv_vfmax_vf_f32m1(vo2p0, params->scalar.min, vl);
      vfloat32m1_t vo3 = __riscv_vfmax_vf_f32m1(vo3p0, params->scalar.min, vl);

      vo0 = __riscv_vfmin_vf_f32m1(vo0, params->scalar.max, vl);
      vo1 = __riscv_vfmin_vf_f32m1(vo1, params->scalar.max, vl);
      vo2 = __riscv_vfmin_vf_f32m1(vo2, params->scalar.max, vl);
      vo3 = __riscv_vfmin_vf_f32m1(vo3, params->scalar.max, vl);

      __riscv_vse32_v_f32m1(o3, vo3, vl); o3 += vl;
      __riscv_vse32_v_f32m1(o2, vo2, vl); o2 += vl;
      __riscv_vse32_v_f32m1(o1, vo1, vl); o1 += vl;
      __riscv_vse32_v_f32m1(o0, vo0, vl); o0 += vl;
    }

    assert(w >= 1 * sizeof(float));
    assert(w <= vl * sizeof(float));
    {
        const vuint32m1_t vmask = __riscv_vle32_v_u32m1((const uint32_t*) &mask_table[(nr-1) - (((input_width >> 2) - 1) & (nr-1))], vl);
        vfloat32m1_t vo0p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
        vfloat32m1_t vo1p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
        vfloat32m1_t vo2p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);
        vfloat32m1_t vo3p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);

        vi0x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi0x4567), vl));
        vi1x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi1x4567), vl));
        vi2x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi2x4567), vl));
        vi3x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi3x4567), vl));
        vi4x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi4x4567), vl));
        vi5x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi5x4567), vl));
        vi6x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi6x4567), vl));
        vi7x4567 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask, __riscv_vreinterpret_v_f32m1_u32m1(vi7x4567), vl));

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[3], vi0x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[3], vi1x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[3], vi2x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[3], vi3x4567, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[8], vi1x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[8], vi2x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[8], vi3x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[8], vi4x4567, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[13], vi2x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[13], vi3x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[13], vi4x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[13], vi5x4567, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[18], vi3x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[18], vi4x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[18], vi5x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[18], vi6x4567, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[23], vi4x4567, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[23], vi5x4567, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[23], vi6x4567, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[23], vi7x4567, vl);

        const vfloat32m1_t vi0x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x0123, vl-1, vl), vi0x4567, 1, vl);
        const vfloat32m1_t vi1x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x0123, vl-1, vl), vi1x4567, 1, vl);
        const vfloat32m1_t vi2x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x0123, vl-1, vl), vi2x4567, 1, vl);
        const vfloat32m1_t vi3x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x0123, vl-1, vl), vi3x4567, 1, vl);
        const vfloat32m1_t vi4x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x0123, vl-1, vl), vi4x4567, 1, vl);
        const vfloat32m1_t vi5x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi5x0123, vl-1, vl), vi5x4567, 1, vl);
        const vfloat32m1_t vi6x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi6x0123, vl-1, vl), vi6x4567, 1, vl);
        const vfloat32m1_t vi7x3456 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi7x0123, vl-1, vl), vi7x4567, 1, vl);


        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[2], vi0x3456, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[2], vi1x3456, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[2], vi2x3456, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[2], vi3x3456, vl);
  
        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[7], vi1x3456, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[7], vi2x3456, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[7], vi3x3456, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[7], vi4x3456, vl);
  
        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[12], vi2x3456, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[12], vi3x3456, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[12], vi4x3456, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[12], vi5x3456, vl);
  
        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[17], vi3x3456, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[17], vi4x3456, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[17], vi5x3456, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[17], vi6x3456, vl);
  
        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[22], vi4x3456, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[22], vi5x3456, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[22], vi6x3456, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[22], vi7x3456, vl);

      const vfloat32m1_t vi0x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x0123, vl-2, vl), vi0x4567, 2, vl);
      const vfloat32m1_t vi1x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x0123, vl-2, vl), vi1x4567, 2, vl);
      const vfloat32m1_t vi2x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x0123, vl-2, vl), vi2x4567, 2, vl);
      const vfloat32m1_t vi3x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x0123, vl-2, vl), vi3x4567, 2, vl);
      const vfloat32m1_t vi4x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x0123, vl-2, vl), vi4x4567, 2, vl);
      const vfloat32m1_t vi5x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi5x0123, vl-2, vl), vi5x4567, 2, vl);
      const vfloat32m1_t vi6x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi6x0123, vl-2, vl), vi6x4567, 2, vl);
      const vfloat32m1_t vi7x2345 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi7x0123, vl-2, vl), vi7x4567, 2, vl);

        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[1], vi0x2345, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[1], vi1x2345, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[1], vi2x2345, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[1], vi3x2345, vl);
  
        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[6], vi1x2345, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[6], vi2x2345, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[6], vi3x2345, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[6], vi4x2345, vl);
  
        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[11], vi2x2345, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[11], vi3x2345, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[11], vi4x2345, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[11], vi5x2345, vl);
  
        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[16], vi3x2345, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[16], vi4x2345, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[16], vi5x2345, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[16], vi6x2345, vl);
  
        vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[21], vi4x2345, vl);
        vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[21], vi5x2345, vl);
        vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[21], vi6x2345, vl);
        vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[21], vi7x2345, vl);

      const vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      const vfloat32m1_t vi0x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x4567, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi1x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x4567, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi2x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x4567, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi3x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x4567, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi4x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x4567, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi5x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi5x4567, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi6x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi6x4567, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi7x5678 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi7x4567, 1, vl), vzero, vl-1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[4], vi0x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[4], vi1x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[4], vi2x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[4], vi3x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[9], vi1x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[9], vi2x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[9], vi3x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[9], vi4x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[14], vi2x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[14], vi3x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[14], vi4x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[14], vi5x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[19], vi3x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[19], vi4x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[19], vi5x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[19], vi6x5678, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[24], vi4x5678, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[24], vi5x5678, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[24], vi6x5678, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[24], vi7x5678, vl);

      const vfloat32m1_t vi0x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x5678, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi1x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x5678, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi2x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x5678, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi3x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x5678, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi4x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x5678, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi5x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi5x5678, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi6x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi6x5678, 1, vl), vzero, vl-1, vl);
      const vfloat32m1_t vi7x6789 = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi7x5678, 1, vl), vzero, vl-1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[5], vi0x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[5], vi1x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[5], vi2x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[5], vi3x6789, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[10], vi1x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[10], vi2x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[10], vi3x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[10], vi4x6789, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[15], vi2x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[15], vi3x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[15], vi4x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[15], vi5x6789, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[20], vi3x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[20], vi4x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[20], vi5x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[20], vi6x6789, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[25], vi4x6789, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, weights[25], vi5x6789, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, weights[25], vi6x6789, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, weights[25], vi7x6789, vl);


      vfloat32m1_t vo0 = __riscv_vfmax_vf_f32m1(vo0p0, params->scalar.min, vl);
      vfloat32m1_t vo1 = __riscv_vfmax_vf_f32m1(vo1p0, params->scalar.min, vl);
      vfloat32m1_t vo2 = __riscv_vfmax_vf_f32m1(vo2p0, params->scalar.min, vl);
      vfloat32m1_t vo3 = __riscv_vfmax_vf_f32m1(vo3p0, params->scalar.min, vl);

      vo0 = __riscv_vfmin_vf_f32m1(vo0, params->scalar.max, vl);
      vo1 = __riscv_vfmin_vf_f32m1(vo1, params->scalar.max, vl);
      vo2 = __riscv_vfmin_vf_f32m1(vo2, params->scalar.max, vl);
      vo3 = __riscv_vfmin_vf_f32m1(vo3, params->scalar.max, vl);

      __riscv_vse32_v_f32m1(o3, vo3, w >> 2); o3 += w >> 2;
      __riscv_vse32_v_f32m1(o2, vo2, w >> 2); o2 += w >> 2;
      __riscv_vse32_v_f32m1(o1, vo1, w >> 2); o1 += w >> 2;
      __riscv_vse32_v_f32m1(o0, vo0, w >> 2); o0 += w >> 2;
    }

    i0 = (const float*) ((uintptr_t) i4 - input_decrement);
    i1 = (const float*) ((uintptr_t) i5 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);

    o0 = o3;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);

    output_height = doz(output_height, 4);
  } while (output_height != 0);
}
