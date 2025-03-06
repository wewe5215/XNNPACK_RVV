#include <assert.h>
#include <riscv_vector.h>
#include "xnnpack/avgpool.h"
void xnn_f32_avgpool_cnhw_minmax_ukernel_7p7x__rvv_c4v(
    size_t kernel_elements/*pooling_size*/,
    size_t total_output,
    const float* input,
    float* output,
    const struct xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(kernel_elements != 0);
  assert(total_output != 0);

  const float scale = params->scalar.scale;
  const float min = params->scalar.min;
  const float max = params->scalar.max;
  float* o = output;
  do {
    int32_t nr = __riscv_vsetvl_e32m4(total_output);
    const float* i0 = input;
    const float* i1 = i0 + nr;
    const float* i2 = i1 + nr;
    const float* i3 = i2 + nr;
    const float* i4 = i3 + nr;
    const float* i5 = i4 + nr;
    const float* i6 = i5 + nr;

    vfloat32m4_t i0_f32v = __riscv_vle32_v_f32m4(i0, nr);
    vfloat32m4_t i1_f32v = __riscv_vle32_v_f32m4(i1, nr);
    vfloat32m4_t i2_f32v = __riscv_vle32_v_f32m4(i2, nr);
    vfloat32m4_t i3_f32v = __riscv_vle32_v_f32m4(i3, nr);
    vfloat32m4_t i4_f32v = __riscv_vle32_v_f32m4(i4, nr);
    vfloat32m4_t i5_f32v = __riscv_vle32_v_f32m4(i5, nr);
    vfloat32m4_t i6_f32v = __riscv_vle32_v_f32m4(i6, nr);

    vfloat32m4_t sum01_f32v = __riscv_vfadd_vv_f32m4(i0_f32v, i1_f32v, nr);
    vfloat32m4_t sum23_f32v = __riscv_vfadd_vv_f32m4(i2_f32v, i3_f32v, nr);
    vfloat32m4_t sum45_f32v = __riscv_vfadd_vv_f32m4(i4_f32v, i5_f32v, nr);
    vfloat32m4_t sum016_f32v = __riscv_vfadd_vv_f32m4(sum01_f32v, i6_f32v, nr);
    vfloat32m4_t sum2345_f32v = __riscv_vfadd_vv_f32m4(sum23_f32v, sum45_f32v, nr);
    vfloat32m4_t sum_f32v = __riscv_vfadd_vv_f32m4(sum2345_f32v, sum016_f32v, nr);
    __riscv_vse32_v_f32m4(o, sum_f32v, nr);
    const float* input_cur = input + 7 * nr;
    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 7; k > 0; k -= 7) {
      const float* i0 = input_cur;
      const float* i1 = i0 + nr;
      const float* i2 = i1 + nr;
      const float* i3 = i2 + nr;
      const float* i4 = i3 + nr;
      const float* i5 = i4 + nr;
      const float* i6 = i5 + nr;
      int32_t nr = __riscv_vsetvl_e32m4(total_output);

      vfloat32m4_t i0_f32v = __riscv_vle32_v_f32m4(i0, nr);
      vfloat32m4_t i1_f32v = __riscv_vle32_v_f32m4(i1, nr);
      vfloat32m4_t i2_f32v = __riscv_vle32_v_f32m4(i2, nr);
      vfloat32m4_t i3_f32v = __riscv_vle32_v_f32m4(i3, nr);
      vfloat32m4_t i4_f32v = __riscv_vle32_v_f32m4(i4, nr);
      vfloat32m4_t i5_f32v = __riscv_vle32_v_f32m4(i5, nr);
      vfloat32m4_t i6_f32v = __riscv_vle32_v_f32m4(i6, nr);
      vfloat32m4_t vacc_f32v = __riscv_vle32_v_f32m4(o, nr);

      vfloat32m4_t sum01_f32v = __riscv_vfadd_vv_f32m4(i0_f32v, i1_f32v, nr);
      vfloat32m4_t sum23_f32v = __riscv_vfadd_vv_f32m4(i2_f32v, i3_f32v, nr);
      vfloat32m4_t sum45_f32v = __riscv_vfadd_vv_f32m4(i4_f32v, i5_f32v, nr);
      vfloat32m4_t sum6a_f32v = __riscv_vfadd_vv_f32m4(i6_f32v, vacc_f32v, nr);
      vfloat32m4_t sum0123_f32v = __riscv_vfadd_vv_f32m4(sum01_f32v, sum23_f32v, nr);
      vfloat32m4_t sum456a_f32v = __riscv_vfadd_vv_f32m4(sum45_f32v, sum6a_f32v, nr);
      vfloat32m4_t sum_f32v = __riscv_vfadd_vv_f32m4(sum0123_f32v, sum456a_f32v, nr);
      __riscv_vse32_v_f32m4(o, sum_f32v, nr);
      input_cur = input_cur + 7 * nr;
    }
    vfloat32m4_t out_f32v = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(o, nr), scale, nr);
    __riscv_vse32_v_f32m4(o, out_f32v, nr);
    input = (const float*) ((uintptr_t) input + (kernel_elements * nr << 2));
    o += nr;
    total_output -= nr;
  } while (total_output != 0);
}
