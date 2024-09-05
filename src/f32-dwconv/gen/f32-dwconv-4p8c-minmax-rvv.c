// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_4p8c__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      vfloat32m1_t vacc01234567p0 = __riscv_vle32_v_f32m1(w, 8); w += 8;

      vfloat32m1_t vi0x01234567 = __riscv_vle32_v_f32m1(i0, 8); i0 += 8;
      vfloat32m1_t vk0x01234567 = __riscv_vle32_v_f32m1(w, 8); w += 8;
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi0x01234567, vk0x01234567, 8);

      vfloat32m1_t vi1x01234567 = __riscv_vle32_v_f32m1(i1, 8); i1 += 8;
      vfloat32m1_t vk1x01234567 = __riscv_vle32_v_f32m1(w, 8); w += 8;
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi1x01234567, vk1x01234567, 8);

      vfloat32m1_t vi2x01234567 = __riscv_vle32_v_f32m1(i2, 8); i2 += 8;
      vfloat32m1_t vk2x01234567 = __riscv_vle32_v_f32m1(w, 8); w += 8;
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi2x01234567, vk2x01234567, 8);

      vfloat32m1_t vi3x01234567 = __riscv_vle32_v_f32m1(i3, 8); i3 += 8;
      vfloat32m1_t vk3x01234567 = __riscv_vle32_v_f32m1(w, 8); w += 8;
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi3x01234567, vk3x01234567, 8);

      vfloat32m1_t vacc01234567 = __riscv_vfmax_vf_f32m1(vacc01234567p0, vmin, 8);
      vacc01234567 = __riscv_vfmin_vf_f32m1(vacc01234567, vmax, 8);
      __riscv_vse32_v_f32m1(output, vacc01234567, 8); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      vfloat32m1_t vacc01234567p0 = __riscv_vle32_v_f32m1(w, c);

      vfloat32m1_t vi0x01234567 = __riscv_vle32_v_f32m1(i0, c);
      vfloat32m1_t vk0x01234567 = __riscv_vle32_v_f32m1(w + 8, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi0x01234567, vk0x01234567, c);

      vfloat32m1_t vi1x01234567 = __riscv_vle32_v_f32m1(i1, c);
      vfloat32m1_t vk1x01234567 = __riscv_vle32_v_f32m1(w + 16, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi1x01234567, vk1x01234567, c);

      vfloat32m1_t vi2x01234567 = __riscv_vle32_v_f32m1(i2, c);
      vfloat32m1_t vk2x01234567 = __riscv_vle32_v_f32m1(w + 24, c);

      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi2x01234567, vk2x01234567, c);

      vfloat32m1_t vi3x01234567 = __riscv_vle32_v_f32m1(i3, c);
      vfloat32m1_t vk3x01234567 = __riscv_vle32_v_f32m1(w + 32, c);

      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi3x01234567, vk3x01234567, c);

      vfloat32m1_t vacc01234567 = __riscv_vfmax_vf_f32m1(vacc01234567p0, vmin, c);
      vacc01234567 = __riscv_vfmin_vf_f32m1(vacc01234567, vmax, c);
      __riscv_vse32_v_f32m1(output, vacc01234567, c); output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
