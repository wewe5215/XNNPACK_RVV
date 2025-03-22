#include <assert.h>
#include <riscv_vector.h>
#include "xnnpack/avgpool.h"
void xnn_f32_avgpool_cnhw_minmax_ukernel_4p4x__rvv_c4v(
    size_t kernel_elements/*pooling_size*/,
    size_t total_output,
    const float* input,
    float* output,
    const struct xnn_f32_scaleminmax_params params[1])
{
  assert(kernel_elements != 0);
  assert(total_output != 0);
  const int nr = __riscv_vsetvlmax_e32m4();
  const float scale = params->scalar.scale;
  const float min = params->scalar.min;
  const float max = params->scalar.max;
  float* o = output;
  int remaining = total_output;
  do {
    int32_t vl = __riscv_vsetvl_e32m4(remaining);
    const float* i0 = input;
    const float* i1 = i0 + nr;
    const float* i2 = i1 + nr;
    const float* i3 = i2 + nr;
    vfloat32m4_t i0_f32v = __riscv_vle32_v_f32m4(i0, vl);
    vfloat32m4_t i1_f32v = __riscv_vle32_v_f32m4(i1, vl);
    vfloat32m4_t i2_f32v = __riscv_vle32_v_f32m4(i2, vl);
    vfloat32m4_t i3_f32v = __riscv_vle32_v_f32m4(i3, vl);

    vfloat32m4_t sum01_f32v = __riscv_vfadd_vv_f32m4(i0_f32v, i1_f32v, vl);
    vfloat32m4_t sum23_f32v = __riscv_vfadd_vv_f32m4(i2_f32v, i3_f32v, vl);
    vfloat32m4_t sum_f32v = __riscv_vfadd_vv_f32m4(sum01_f32v, sum23_f32v, vl);
    __riscv_vse32_v_f32m4(o, sum_f32v, vl);
    const float* input_cur = input + 4 * nr;
    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 4; k > 0; k -= 4) {
      const float* i0 = input_cur;
      const float* i1 = i0 + nr;
      const float* i2 = i1 + nr;
      const float* i3 = i2 + nr;
      int32_t vl = __riscv_vsetvl_e32m4(total_output);

      vfloat32m4_t i0_f32v = __riscv_vle32_v_f32m4(i0, vl);
      vfloat32m4_t i1_f32v = __riscv_vle32_v_f32m4(i1, vl);
      vfloat32m4_t i2_f32v = __riscv_vle32_v_f32m4(i2, vl);
      vfloat32m4_t i3_f32v = __riscv_vle32_v_f32m4(i3, vl);
      vfloat32m4_t vacc_f32v = __riscv_vle32_v_f32m4(o, vl);

      vfloat32m4_t sum01_f32v = __riscv_vfadd_vv_f32m4(i0_f32v, i1_f32v, vl);
      vfloat32m4_t sum23_f32v = __riscv_vfadd_vv_f32m4(i2_f32v, i3_f32v, vl);
      vfloat32m4_t sum0123_f32v = __riscv_vfadd_vv_f32m4(sum01_f32v, sum23_f32v, vl);
      vfloat32m4_t sum_f32v = __riscv_vfadd_vv_f32m4(sum0123_f32v, vacc_f32v, vl);
      __riscv_vse32_v_f32m4(o, sum_f32v, vl);
      input_cur = input_cur + 4 * vl;
    }
    vfloat32m4_t out_f32v = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(o, vl), scale, vl);
    __riscv_vse32_v_f32m4(o, out_f32v, vl);
    input = (const float*) ((uintptr_t) input + (kernel_elements * nr << 2));
    o += nr;
    remaining -= nr;
  } while (remaining > 0);
}
