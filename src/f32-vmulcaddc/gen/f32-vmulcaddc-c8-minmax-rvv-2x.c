// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/math.h"
#include "xnnpack/vmulcaddc.h"


void xnn_f32_vmulcaddc_minmax_ukernel_c8__rvv_2x(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const vfloat32m1_t vscale01234567 = __riscv_vle32_v_f32m1(w, 8); w += 8;

      vfloat32m1_t vacc0x01234567 = __riscv_vle32_v_f32m1(i0, 8); i0 += 8;
      vfloat32m1_t vacc1x01234567 = __riscv_vle32_v_f32m1(i1, 8); i1 += 8;

      vacc0x01234567 = __riscv_vfmul_vv_f32m1(vacc0x01234567, vscale01234567, 8);
      vacc1x01234567 = __riscv_vfmul_vv_f32m1(vacc1x01234567, vscale01234567, 8);

      const vfloat32m1_t vbias01234567 = __riscv_vle32_v_f32m1(w, 8); w += 8;

      vacc0x01234567 = __riscv_vfadd_vv_f32m1(vacc0x01234567, vbias01234567, 8);
      vacc1x01234567 = __riscv_vfadd_vv_f32m1(vacc1x01234567, vbias01234567, 8);

      vacc0x01234567 = __riscv_vfmax_vf_f32m1(vacc0x01234567, vmin, 8);
      vacc1x01234567 = __riscv_vfmax_vf_f32m1(vacc1x01234567, vmin, 8);

      vacc0x01234567 = __riscv_vfmin_vf_f32m1(vacc0x01234567, vmax, 8);
      vacc1x01234567 = __riscv_vfmin_vf_f32m1(vacc1x01234567, vmax, 8);

      __riscv_vse32_v_f32m1(o0, vacc0x01234567, 8); o0 += 8;
      __riscv_vse32_v_f32m1(o1, vacc1x01234567, 8); o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t n = c / sizeof(float);
      const vfloat32m1_t vscale0123 = __riscv_vle32_v_f32m1(w, n);

      vfloat32m1_t vacc0x0123 = __riscv_vle32_v_f32m1(i0, n); i0 = (const float*) ((uintptr_t) i0 + c);
      vfloat32m1_t vacc1x0123 = __riscv_vle32_v_f32m1(i1, n); i1 = (const float*) ((uintptr_t) i1 + c);

      vacc0x0123 = __riscv_vfmul_vv_f32m1(vacc0x0123, vscale0123, n);
      vacc1x0123 = __riscv_vfmul_vv_f32m1(vacc1x0123, vscale0123, n);

      const vfloat32m1_t vbias0123 = __riscv_vle32_v_f32m1(w + 8, n);

      vacc0x0123 = __riscv_vfadd_vv_f32m1(vacc0x0123, vbias0123, n);
      vacc1x0123 = __riscv_vfadd_vv_f32m1(vacc1x0123, vbias0123, n);

      vacc0x0123 = __riscv_vfmax_vf_f32m1(vacc0x0123, vmin, n);
      vacc1x0123 = __riscv_vfmax_vf_f32m1(vacc1x0123, vmin, n);

      vacc0x0123 = __riscv_vfmin_vf_f32m1(vacc0x0123, vmax, n);
      vacc1x0123 = __riscv_vfmin_vf_f32m1(vacc1x0123, vmax, n);

      __riscv_vse32_v_f32m1(o0, vacc0x0123, n); o0 += n;
      __riscv_vse32_v_f32m1(o1, vacc1x0123, n); o1 += n;
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
