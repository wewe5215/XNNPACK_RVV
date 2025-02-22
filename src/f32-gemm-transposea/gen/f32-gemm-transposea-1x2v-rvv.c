// Auto-generated file. Do not edit!
//   Template: src/f32-gemm-transposea/MRxNRv-rvv.c.in
//   Generator: tools/xngen
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/gemm.h"


void xnn_f32_transpose_a_gemm_ukernel_1x2v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t w_stride,
    const float*  w,
    const float*  bias,
    float*  c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(bias != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* w0 = w;
  float* c0 = c;
  const float* bias0 = bias;

  const size_t nr = __riscv_vsetvlmax_e32m2();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m2(nc);
    }
    nc = nc - vl;

    vfloat32m2_t vacc0 =  __riscv_vfmv_v_f_f32m2(bias0, vl);

    size_t k = kc;
    do {
      const float vw0 = *w0++;
      vfloat32m2_t vb = __riscv_vle32_v_f32m2(a, vl);
      a = a + nr;
      vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, vw0, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // store 1 x vl results to c
    __riscv_vse32_v_f32m2(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    w0 = (const float*) ((uintptr_t) w0 - kc);
  } while (nc != 0);
}
