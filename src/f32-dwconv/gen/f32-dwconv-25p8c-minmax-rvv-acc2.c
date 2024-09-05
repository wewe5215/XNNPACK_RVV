// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_25p8c__rvv_acc2(
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
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
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
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      vfloat32m1_t vacc01234567p0 = __riscv_vle32_v_f32m1(w, 8);


      const vfloat32m1_t vi0x01234567 = __riscv_vle32_v_f32m1(i0, 8);
      i0 += 8;

      const vfloat32m1_t vk0x01234567 = __riscv_vle32_v_f32m1(w + 8, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi0x01234567, vk0x01234567, 8);

      const vfloat32m1_t vi1x01234567 = __riscv_vle32_v_f32m1(i1, 8);
      i1 += 8;

      const vfloat32m1_t vk1x01234567 = __riscv_vle32_v_f32m1(w + 16, 8);
      vfloat32m1_t vacc01234567p1 = __riscv_vfmul_vv_f32m1(vi1x01234567, vk1x01234567, 8);

      const vfloat32m1_t vi2x01234567 = __riscv_vle32_v_f32m1(i2, 8);
      i2 += 8;

      const vfloat32m1_t vk2x01234567 = __riscv_vle32_v_f32m1(w + 24, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi2x01234567, vk2x01234567, 8);

      const vfloat32m1_t vi3x01234567 = __riscv_vle32_v_f32m1(i3, 8);
      i3 += 8;

      const vfloat32m1_t vk3x01234567 = __riscv_vle32_v_f32m1(w + 32, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi3x01234567, vk3x01234567, 8);

      const vfloat32m1_t vi4x01234567 = __riscv_vle32_v_f32m1(i4, 8);
      i4 += 8;

      const vfloat32m1_t vk4x01234567 = __riscv_vle32_v_f32m1(w + 40, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi4x01234567, vk4x01234567, 8);

      const vfloat32m1_t vi5x01234567 = __riscv_vle32_v_f32m1(i5, 8);
      i5 += 8;

      const vfloat32m1_t vk5x01234567 = __riscv_vle32_v_f32m1(w + 48, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi5x01234567, vk5x01234567, 8);

      const vfloat32m1_t vi6x01234567 = __riscv_vle32_v_f32m1(i6, 8);
      i6 += 8;

      const vfloat32m1_t vk6x01234567 = __riscv_vle32_v_f32m1(w + 56, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi6x01234567, vk6x01234567, 8);

      const vfloat32m1_t vi7x01234567 = __riscv_vle32_v_f32m1(i7, 8);
      i7 += 8;

      const vfloat32m1_t vk7x01234567 = __riscv_vle32_v_f32m1(w + 64, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi7x01234567, vk7x01234567, 8);

      const vfloat32m1_t vi8x01234567 = __riscv_vle32_v_f32m1(i8, 8);
      i8 += 8;

      const vfloat32m1_t vk8x01234567 = __riscv_vle32_v_f32m1(w + 72, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi8x01234567, vk8x01234567, 8);

      const vfloat32m1_t vi9x01234567 = __riscv_vle32_v_f32m1(i9, 8);
      i9 += 8;

      const vfloat32m1_t vk9x01234567 = __riscv_vle32_v_f32m1(w + 80, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi9x01234567, vk9x01234567, 8);

      const vfloat32m1_t vi10x01234567 = __riscv_vle32_v_f32m1(i10, 8);
      i10 += 8;

      const vfloat32m1_t vk10x01234567 = __riscv_vle32_v_f32m1(w + 88, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi10x01234567, vk10x01234567, 8);

      const vfloat32m1_t vi11x01234567 = __riscv_vle32_v_f32m1(i11, 8);
      i11 += 8;

      const vfloat32m1_t vk11x01234567 = __riscv_vle32_v_f32m1(w + 96, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi11x01234567, vk11x01234567, 8);

      const vfloat32m1_t vi12x01234567 = __riscv_vle32_v_f32m1(i12, 8);
      i12 += 8;

      const vfloat32m1_t vk12x01234567 = __riscv_vle32_v_f32m1(w + 104, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi12x01234567, vk12x01234567, 8);

      const vfloat32m1_t vi13x01234567 = __riscv_vle32_v_f32m1(i13, 8);
      i13 += 8;

      const vfloat32m1_t vk13x01234567 = __riscv_vle32_v_f32m1(w + 112, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi13x01234567, vk13x01234567, 8);

      const vfloat32m1_t vi14x01234567 = __riscv_vle32_v_f32m1(i14, 8);
      i14 += 8;

      const vfloat32m1_t vk14x01234567 = __riscv_vle32_v_f32m1(w + 120, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi14x01234567, vk14x01234567, 8);

      const vfloat32m1_t vi15x01234567 = __riscv_vle32_v_f32m1(i15, 8);
      i15 += 8;

      const vfloat32m1_t vk15x01234567 = __riscv_vle32_v_f32m1(w + 128, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi15x01234567, vk15x01234567, 8);

      const vfloat32m1_t vi16x01234567 = __riscv_vle32_v_f32m1(i16, 8);
      i16 += 8;

      const vfloat32m1_t vk16x01234567 = __riscv_vle32_v_f32m1(w + 136, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi16x01234567, vk16x01234567, 8);

      const vfloat32m1_t vi17x01234567 = __riscv_vle32_v_f32m1(i17, 8);
      i17 += 8;

      const vfloat32m1_t vk17x01234567 = __riscv_vle32_v_f32m1(w + 144, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi17x01234567, vk17x01234567, 8);

      const vfloat32m1_t vi18x01234567 = __riscv_vle32_v_f32m1(i18, 8);
      i18 += 8;

      const vfloat32m1_t vk18x01234567 = __riscv_vle32_v_f32m1(w + 152, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi18x01234567, vk18x01234567, 8);

      const vfloat32m1_t vi19x01234567 = __riscv_vle32_v_f32m1(i19, 8);
      i19 += 8;

      const vfloat32m1_t vk19x01234567 = __riscv_vle32_v_f32m1(w + 160, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi19x01234567, vk19x01234567, 8);

      const vfloat32m1_t vi20x01234567 = __riscv_vle32_v_f32m1(i20, 8);
      i20 += 8;

      const vfloat32m1_t vk20x01234567 = __riscv_vle32_v_f32m1(w + 168, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi20x01234567, vk20x01234567, 8);

      const vfloat32m1_t vi21x01234567 = __riscv_vle32_v_f32m1(i21, 8);
      i21 += 8;

      const vfloat32m1_t vk21x01234567 = __riscv_vle32_v_f32m1(w + 176, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi21x01234567, vk21x01234567, 8);

      const vfloat32m1_t vi22x01234567 = __riscv_vle32_v_f32m1(i22, 8);
      i22 += 8;

      const vfloat32m1_t vk22x01234567 = __riscv_vle32_v_f32m1(w + 184, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi22x01234567, vk22x01234567, 8);

      const vfloat32m1_t vi23x01234567 = __riscv_vle32_v_f32m1(i23, 8);
      i23 += 8;

      const vfloat32m1_t vk23x01234567 = __riscv_vle32_v_f32m1(w + 192, 8);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi23x01234567, vk23x01234567, 8);

      const vfloat32m1_t vi24x01234567 = __riscv_vle32_v_f32m1(i24, 8);
      i24 += 8;

      const vfloat32m1_t vk24x01234567 = __riscv_vle32_v_f32m1(w + 200, 8);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi24x01234567, vk24x01234567, 8);

      w += 208;

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = __riscv_vfadd_vv_f32m1(vacc01234567p0, vacc01234567p1, 8);

      vfloat32m1_t vacc01234567 = __riscv_vfmax_vf_f32m1(vacc01234567p0, vmin, 8);
      vacc01234567 = __riscv_vfmin_vf_f32m1(vacc01234567, vmax, 8);

      __riscv_vse32_v_f32m1(output, vacc01234567, 8);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      vfloat32m1_t vacc01234567p0 = __riscv_vle32_v_f32m1(w, c);

      const vfloat32m1_t vi0x01234567 = __riscv_vle32_v_f32m1(i0, c);
      const vfloat32m1_t vk0x01234567 = __riscv_vle32_v_f32m1(w + 8, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi0x01234567, vk0x01234567, c);

      const vfloat32m1_t vi1x01234567 = __riscv_vle32_v_f32m1(i1, c);
      const vfloat32m1_t vk1x01234567 = __riscv_vle32_v_f32m1(w + 16, c);
      vfloat32m1_t vacc01234567p1 = __riscv_vfmul_vv_f32m1(vi1x01234567, vk1x01234567, c);

      const vfloat32m1_t vi2x01234567 = __riscv_vle32_v_f32m1(i2, c);
      const vfloat32m1_t vk2x01234567 = __riscv_vle32_v_f32m1(w + 24, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi2x01234567, vk2x01234567, c);

      const vfloat32m1_t vi3x01234567 = __riscv_vle32_v_f32m1(i3, c);
      const vfloat32m1_t vk3x01234567 = __riscv_vle32_v_f32m1(w + 32, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi3x01234567, vk3x01234567, c);

      const vfloat32m1_t vi4x01234567 = __riscv_vle32_v_f32m1(i4, c);
      const vfloat32m1_t vk4x01234567 = __riscv_vle32_v_f32m1(w + 40, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi4x01234567, vk4x01234567, c);

      const vfloat32m1_t vi5x01234567 = __riscv_vle32_v_f32m1(i5, c);
      const vfloat32m1_t vk5x01234567 = __riscv_vle32_v_f32m1(w + 48, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi5x01234567, vk5x01234567, c);

      const vfloat32m1_t vi6x01234567 = __riscv_vle32_v_f32m1(i6, c);
      const vfloat32m1_t vk6x01234567 = __riscv_vle32_v_f32m1(w + 56, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi6x01234567, vk6x01234567, c);

      const vfloat32m1_t vi7x01234567 = __riscv_vle32_v_f32m1(i7, c);
      const vfloat32m1_t vk7x01234567 = __riscv_vle32_v_f32m1(w + 64, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi7x01234567, vk7x01234567, c);

      const vfloat32m1_t vi8x01234567 = __riscv_vle32_v_f32m1(i8, c);
      const vfloat32m1_t vk8x01234567 = __riscv_vle32_v_f32m1(w + 72, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi8x01234567, vk8x01234567, c);

      const vfloat32m1_t vi9x01234567 = __riscv_vle32_v_f32m1(i9, c);
      const vfloat32m1_t vk9x01234567 = __riscv_vle32_v_f32m1(w + 80, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi9x01234567, vk9x01234567, c);

      const vfloat32m1_t vi10x01234567 = __riscv_vle32_v_f32m1(i10, c);
      const vfloat32m1_t vk10x01234567 = __riscv_vle32_v_f32m1(w + 88, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi10x01234567, vk10x01234567, c);

      const vfloat32m1_t vi11x01234567 = __riscv_vle32_v_f32m1(i11, c);
      const vfloat32m1_t vk11x01234567 = __riscv_vle32_v_f32m1(w + 96, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi11x01234567, vk11x01234567, c);

      const vfloat32m1_t vi12x01234567 = __riscv_vle32_v_f32m1(i12, c);
      const vfloat32m1_t vk12x01234567 = __riscv_vle32_v_f32m1(w + 104, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi12x01234567, vk12x01234567, c);

      const vfloat32m1_t vi13x01234567 = __riscv_vle32_v_f32m1(i13, c);
      const vfloat32m1_t vk13x01234567 = __riscv_vle32_v_f32m1(w + 112, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi13x01234567, vk13x01234567, c);

      const vfloat32m1_t vi14x01234567 = __riscv_vle32_v_f32m1(i14, c);
      const vfloat32m1_t vk14x01234567 = __riscv_vle32_v_f32m1(w + 120, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi14x01234567, vk14x01234567, c);

      const vfloat32m1_t vi15x01234567 = __riscv_vle32_v_f32m1(i15, c);
      const vfloat32m1_t vk15x01234567 = __riscv_vle32_v_f32m1(w + 128, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi15x01234567, vk15x01234567, c);

      const vfloat32m1_t vi16x01234567 = __riscv_vle32_v_f32m1(i16, c);
      const vfloat32m1_t vk16x01234567 = __riscv_vle32_v_f32m1(w + 136, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi16x01234567, vk16x01234567, c);

      const vfloat32m1_t vi17x01234567 = __riscv_vle32_v_f32m1(i17, c);
      const vfloat32m1_t vk17x01234567 = __riscv_vle32_v_f32m1(w + 144, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi17x01234567, vk17x01234567, c);

      const vfloat32m1_t vi18x01234567 = __riscv_vle32_v_f32m1(i18, c);
      const vfloat32m1_t vk18x01234567 = __riscv_vle32_v_f32m1(w + 152, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi18x01234567, vk18x01234567, c);

      const vfloat32m1_t vi19x01234567 = __riscv_vle32_v_f32m1(i19, c);
      const vfloat32m1_t vk19x01234567 = __riscv_vle32_v_f32m1(w + 160, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi19x01234567, vk19x01234567, c);

      const vfloat32m1_t vi20x01234567 = __riscv_vle32_v_f32m1(i20, c);
      const vfloat32m1_t vk20x01234567 = __riscv_vle32_v_f32m1(w + 168, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi20x01234567, vk20x01234567, c);

      const vfloat32m1_t vi21x01234567 = __riscv_vle32_v_f32m1(i21, c);
      const vfloat32m1_t vk21x01234567 = __riscv_vle32_v_f32m1(w + 176, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi21x01234567, vk21x01234567, c);

      const vfloat32m1_t vi22x01234567 = __riscv_vle32_v_f32m1(i22, c);
      const vfloat32m1_t vk22x01234567 = __riscv_vle32_v_f32m1(w + 184, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi22x01234567, vk22x01234567, c);

      const vfloat32m1_t vi23x01234567 = __riscv_vle32_v_f32m1(i23, c);
      const vfloat32m1_t vk23x01234567 = __riscv_vle32_v_f32m1(w + 192, c);
      vacc01234567p1 = __riscv_vfmacc_vv_f32m1(vacc01234567p1, vi23x01234567, vk23x01234567, c);

      const vfloat32m1_t vi24x01234567 = __riscv_vle32_v_f32m1(i24, c);
      const vfloat32m1_t vk24x01234567 = __riscv_vle32_v_f32m1(w + 200, c);
      vacc01234567p0 = __riscv_vfmacc_vv_f32m1(vacc01234567p0, vi24x01234567, vk24x01234567, c);

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = __riscv_vfadd_vv_f32m1(vacc01234567p0, vacc01234567p1, c);

      vfloat32m1_t vacc01234567 = __riscv_vfmax_vf_f32m1(vacc01234567p0, vmin, c);
      vacc01234567 = __riscv_vfmin_vf_f32m1(vacc01234567, vmax, c);

      __riscv_vse32_v_f32m1(output, vacc01234567, c); output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
