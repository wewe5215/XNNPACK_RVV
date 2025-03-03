// Auto-generated file. Do not edit!
//   Template: src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in
//   Generator: tools/xngen
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/gemm.h"


void xnn_f32_transpose_a_pruned_gemm_relu_ukernel_3x8v__rvv(
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
    uint16_t* indice,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(bias != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = 0.0f;
  const float* w0 = w;
  float* c0 = c;
  const float* bias0 = bias;
  const float* w1 = (const float*) ((uintptr_t) w0 + w_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  const float* bias1 = (const float*) ((uintptr_t)bias0 + sizeof(float));
  if XNN_UNPREDICTABLE(mr < 2) {
    w1 = w0;
    c1 = c0;
    bias1 = bias0;
  }
  const float* w2 = (const float*) ((uintptr_t) w1 + w_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  const float* bias2 = (const float*) ((uintptr_t)bias1 + sizeof(float));
  if XNN_UNPREDICTABLE(mr <= 2) {
    w2 = w1;
    c2 = c1;
    bias2 = bias1;
  }

  const size_t nr = __riscv_vsetvlmax_e32m8();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m8(nc);
    }
    nc = nc - vl;

    vfloat32m8_t vacc0 =  __riscv_vfmv_v_f_f32m8(*bias0, vl);
    vfloat32m8_t vacc1 =  __riscv_vfmv_v_f_f32m8(*bias1, vl);
    vfloat32m8_t vacc2 =  __riscv_vfmv_v_f_f32m8(*bias2, vl);

    size_t k = kc;
    size_t idx_indice_arr = 0;
    do {
      const float vw0 = *w0++;
      const float vw1 = *w1++;
      const float vw2 = *w2++;
      vfloat32m8_t vb = __riscv_vle32_v_f32m8(a + indice[idx_indice_arr], vl);
      idx_indice_arr++;
      vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, vw0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, vw1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, vw2, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // apply ReLU to results
    vacc0 = __riscv_vfmax_vf_f32m8(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m8(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m8(vacc2, vmin, vl);
    // store 3 x vl results to c
    __riscv_vse32_v_f32m8(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse32_v_f32m8(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse32_v_f32m8(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    w0 = (const float*) ((uintptr_t) w0 - kc);
    w1 = (const float*) ((uintptr_t) w1 - kc);
    w2 = (const float*) ((uintptr_t) w2 - kc);
  } while (nc != 0);
}
