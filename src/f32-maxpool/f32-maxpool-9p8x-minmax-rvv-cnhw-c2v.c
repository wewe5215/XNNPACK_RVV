#include <assert.h>
#include "xnnpack/maxpool.h"
#include <riscv_vector.h>

void xnn_f32_maxpool_cnhw_minmax_ukernel_9p8x__rvv_c2v(
    size_t kernel_elements,
    size_t total_output,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(kernel_elements != 0);
  assert(total_output != 0);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  float* o = output;
  do {
    int32_t nr = __riscv_vsetvl_e32m2(total_output);
    const float* i0 = input;
    const float* i1 = i0 + nr;
    const float* i2 = i1 + nr;
    const float* i3 = i2 + nr;
    const float* i4 = i3 + nr;
    const float* i5 = i4 + nr;
    const float* i6 = i5 + nr;
    const float* i7 = i6 + nr;
    const float* i8 = i7 + nr;
    if (kernel_elements < 2) {
      i1 = i0;
    }
    if (kernel_elements <= 2) {
      i2 = i0;
    }
    if (kernel_elements < 4) {
      i3 = i0;
    }
    if (kernel_elements <= 4) {
      i4 = i0;
    }
    if (kernel_elements < 6) {
      i5 = i0;
    }
    if (kernel_elements <= 6) {
      i6 = i0;
    }
    if (kernel_elements < 8) {
      i7 = i0;
    }
    if (kernel_elements <= 8) {
      i8 = i0;
    }
    vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, nr);
    vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, nr);
    vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, nr);
    vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, nr);
    vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, nr);
    vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, nr);
    vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, nr);
    vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i7, nr);
    vfloat32m2_t i8_f32v = __riscv_vle32_v_f32m2(i8, nr);

    vfloat32m2_t max01_f32v = __riscv_vfmax_vv_f32m2(i0_f32v, i1_f32v, nr);
    vfloat32m2_t max23_f32v = __riscv_vfmax_vv_f32m2(i2_f32v, i3_f32v, nr);
    vfloat32m2_t max45_f32v = __riscv_vfmax_vv_f32m2(i4_f32v, i5_f32v, nr);
    vfloat32m2_t max67_f32v = __riscv_vfmax_vv_f32m2(i6_f32v, i7_f32v, nr);
    vfloat32m2_t max018_f32v = __riscv_vfmax_vv_f32m2(max01_f32v, i8_f32v, nr);

    vfloat32m2_t max2345_f32v = __riscv_vfmax_vv_f32m2(max23_f32v, max45_f32v, nr);
    vfloat32m2_t max01678_f32v = __riscv_vfmax_vv_f32m2(max67_f32v, max018_f32v, nr);
    vfloat32m2_t out_f32v = __riscv_vfmax_vv_f32m2(max2345_f32v, max01678_f32v, nr);
    out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, output_min, nr), output_max, nr);
    __riscv_vse32_v_f32m2(o, out_f32v, nr);
    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const float* i0 = input + 9 * nr;
      const float* i1 = i0 + nr;
      const float* i2 = i1 + nr;
      const float* i3 = i2 + nr;
      const float* i4 = i3 + nr;
      const float* i5 = i4 + nr;
      const float* i6 = i5 + nr;
      const float* i7 = i6 + nr;
      const float* i8 = i7 + nr;
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }
      int32_t nr = __riscv_vsetvl_e32m2(total_output);

      vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, nr);
      vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, nr);
      vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, nr);
      vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, nr);
      vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, nr);
      vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, nr);
      vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, nr);
      vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i7, nr);
      vfloat32m2_t i8_f32v = __riscv_vle32_v_f32m2(o, nr);

      vfloat32m2_t max01_f32v = __riscv_vfmax_vv_f32m2(i0_f32v, i1_f32v, nr);
      vfloat32m2_t max23_f32v = __riscv_vfmax_vv_f32m2(i2_f32v, i3_f32v, nr);
      vfloat32m2_t max45_f32v = __riscv_vfmax_vv_f32m2(i4_f32v, i5_f32v, nr);
      vfloat32m2_t max67_f32v = __riscv_vfmax_vv_f32m2(i6_f32v, i7_f32v, nr);
      vfloat32m2_t max018_f32v = __riscv_vfmax_vv_f32m2(max01_f32v, i8_f32v, nr);

      vfloat32m2_t max2345_f32v = __riscv_vfmax_vv_f32m2(max23_f32v, max45_f32v, nr);
      vfloat32m2_t max01678_f32v = __riscv_vfmax_vv_f32m2(max67_f32v, max018_f32v, nr);
      vfloat32m2_t out_f32v = __riscv_vfmax_vv_f32m2(max2345_f32v, max01678_f32v, nr);
      out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, output_min, nr), output_max, nr);
      __riscv_vse32_v_f32m2(o, out_f32v, nr);
    }
    input = (const float*) ((uintptr_t) input + (kernel_elements * nr << 2));
    o += nr;
    total_output -= nr;
  } while (total_output != 0);
}
