// Auto-generated file. Do not edit!
//   Template: src/f32-gemm-transposea/MRxNRv-rvv.c.in
//   Generator: tools/xngen
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/gemm.h"


void xnn_f32_transpose_a_gemm_relu_ukernel_7x4v__rvv(
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
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
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
  const float* w3 = (const float*) ((uintptr_t) w2 + w_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  const float* bias3 = (const float*) ((uintptr_t)bias2 + sizeof(float));
  if XNN_UNPREDICTABLE(mr < 4) {
    w3 = w2;
    c3 = c2;
    bias3 = bias2;
  }
  const float* w4 = (const float*) ((uintptr_t) w3 + w_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  const float* bias4 = (const float*) ((uintptr_t)bias3 + sizeof(float));
  if XNN_UNPREDICTABLE(mr <= 4) {
    w4 = w3;
    c4 = c3;
    bias4 = bias3;
  }
  const float* w5 = (const float*) ((uintptr_t) w4 + w_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  const float* bias5 = (const float*) ((uintptr_t)bias4 + sizeof(float));
  if XNN_UNPREDICTABLE(mr < 6) {
    w5 = w4;
    c5 = c4;
    bias5 = bias4;
  }
  const float* w6 = (const float*) ((uintptr_t) w5 + w_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  const float* bias6 = (const float*) ((uintptr_t)bias5 + sizeof(float));
  if XNN_UNPREDICTABLE(mr <= 6) {
    w6 = w5;
    c6 = c5;
    bias6 = bias5;
  }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vfloat32m4_t vacc0 =  __riscv_vfmv_v_f_f32m4(bias0, vl);
    vfloat32m4_t vacc1 =  __riscv_vfmv_v_f_f32m4(bias1, vl);
    vfloat32m4_t vacc2 =  __riscv_vfmv_v_f_f32m4(bias2, vl);
    vfloat32m4_t vacc3 =  __riscv_vfmv_v_f_f32m4(bias3, vl);
    vfloat32m4_t vacc4 =  __riscv_vfmv_v_f_f32m4(bias4, vl);
    vfloat32m4_t vacc5 =  __riscv_vfmv_v_f_f32m4(bias5, vl);
    vfloat32m4_t vacc6 =  __riscv_vfmv_v_f_f32m4(bias6, vl);

    size_t k = kc;
    do {
      const float vw0 = *w0++;
      const float vw1 = *w1++;
      const float vw2 = *w2++;
      const float vw3 = *w3++;
      const float vw4 = *w4++;
      const float vw5 = *w5++;
      const float vw6 = *w6++;
      vfloat32m4_t vb = __riscv_vle32_v_f32m4(a, vl);
      a = a + nr;
      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, vw0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, vw1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, vw2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, vw3, vb, vl);
      vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, vw4, vb, vl);
      vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, vw5, vb, vl);
      vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, vw6, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // apply ReLU to results
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc4 = __riscv_vfmax_vf_f32m4(vacc4, vmin, vl);
    vacc5 = __riscv_vfmax_vf_f32m4(vacc5, vmin, vl);
    vacc6 = __riscv_vfmax_vf_f32m4(vacc6, vmin, vl);
    // store 7 x vl results to c
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    __riscv_vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    __riscv_vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    __riscv_vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    w0 = (const float*) ((uintptr_t) w0 - kc);
    w1 = (const float*) ((uintptr_t) w1 - kc);
    w2 = (const float*) ((uintptr_t) w2 - kc);
    w3 = (const float*) ((uintptr_t) w3 - kc);
    w4 = (const float*) ((uintptr_t) w4 - kc);
    w5 = (const float*) ((uintptr_t) w5 - kc);
    w6 = (const float*) ((uintptr_t) w6 - kc);
  } while (nc != 0);
}
