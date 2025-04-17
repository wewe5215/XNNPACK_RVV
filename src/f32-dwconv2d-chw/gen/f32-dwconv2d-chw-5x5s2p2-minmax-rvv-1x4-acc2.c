// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5s2p2-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__rvv_1x4_acc2(
  size_t input_height,
  size_t input_width,
  const float* input,
  const float* weights,
  const float* zero,
  float* output,
  uint32_t padding_top,
  const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
assert(input_height != 0);
assert(input_width != 0);
assert(input_width % sizeof(float) == 0);
assert(padding_top >= 1);
assert(padding_top <= 2);

const size_t nr = __riscv_vsetvlmax_e32m1();
size_t vl = nr;
int32_t mask[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
const vuint32m1_t vmask_even = __riscv_vle32_v_u32m1((const uint32_t*) &mask[(nr-1) - (((input_width - nr) & ((vl << 3) - 1)) >> 3)], vl);
const vuint32m1_t vmask_odd = __riscv_vle32_v_u32m1((const uint32_t*) &mask[nr - ((((input_width - nr) & ((vl << 3) - 1)) + nr) >> 3)], vl);


const uint32_t padding_top_less_1 = padding_top - 1;
const size_t input_decrement = round_up_po2(input_width, vl << 3);

const float* i0 = zero;
const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
  i1 = zero;
}
const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
const float* i4 = (const float*) ((uintptr_t) i3 + input_width);


float* o0 = output;

size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
do {
  if XNN_UNPREDICTABLE(padded_input_height < 6) {
    i3 = zero;
  }
  if XNN_UNPREDICTABLE(padded_input_height < 7) {
    i4 = zero;
  }
  vfloat32m1_t vi0x0246 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi1x0246 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi2x0246 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi3x0246 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi4x0246 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

  vfloat32m1_t vi0x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi1x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi2x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi3x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vi4x1357 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

  vfloat32m1x2_t vi0x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i0, vl); i0 += vl << 1;
  vfloat32m1x2_t vi1x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i1, vl); i1 += vl << 1;
  vfloat32m1x2_t vi2x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i2, vl); i2 += vl << 1;
  vfloat32m1x2_t vi3x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i3, vl); i3 += vl << 1;
  vfloat32m1x2_t vi4x8ACE9BDF = __riscv_vlseg2e32_v_f32m1x2(i4, vl); i4 += vl << 1;

  size_t w = input_width;
  for (; w > vl << 3; w -= vl << 3) {
    vfloat32m1_t vo0p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);

    vfloat32m1_t vo0p1 = __riscv_vfmul_vf_f32m1(__riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 0), weights[3], vl);
    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[8], __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 0), vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[13], __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 0), vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[18], __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 0), vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[23], __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 0), vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[4], __riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 1), vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[9], __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 1), vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[14], __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1), vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[19], __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 1), vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[24], __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 1), vl);

    const vfloat32m1_t vi0x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x0246, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 0), 1, vl);
    vi0x0246 = __riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 0);
    const vfloat32m1_t vi1x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x0246, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 0), 1, vl);
    vi1x0246 = __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 0);
    const vfloat32m1_t vi2x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x0246, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 0), 1, vl);
    vi2x0246 = __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 0);
    const vfloat32m1_t vi3x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x0246, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 0), 1, vl);
    vi3x0246 = __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 0);
    const vfloat32m1_t vi4x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x0246, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 0), 1, vl);
    vi4x0246 = __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 0);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[1], vi0x68AC, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[6], vi1x68AC, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[11], vi2x68AC, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[16], vi3x68AC, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[21], vi4x68AC, vl);

    const vfloat32m1_t vi0x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x1357, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 1), 1, vl);
    vi0x1357 = __riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 1);
    const vfloat32m1_t vi1x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x1357, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 1), 1, vl);
    vi1x1357 = __riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 1);
    const vfloat32m1_t vi2x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x1357, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1), 1, vl);
    vi2x1357 = __riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1);
    const vfloat32m1_t vi3x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x1357, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 1), 1, vl);
    vi3x1357 = __riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 1);
    const vfloat32m1_t vi4x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x1357, vl-1, vl), __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 1), 1, vl);
    vi4x1357 = __riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 1);

    const vfloat32m1x2_t vi0xGIKMHJLN = __riscv_vlseg2e32_v_f32m1x2(i0, vl); i0 += vl << 1;
    const vfloat32m1x2_t vi1xGIKMHJLN = __riscv_vlseg2e32_v_f32m1x2(i1, vl); i1 += vl << 1;
    const vfloat32m1x2_t vi2xGIKMHJLN = __riscv_vlseg2e32_v_f32m1x2(i2, vl); i2 += vl << 1;
    const vfloat32m1x2_t vi3xGIKMHJLN = __riscv_vlseg2e32_v_f32m1x2(i3, vl); i3 += vl << 1;
    const vfloat32m1x2_t vi4xGIKMHJLN = __riscv_vlseg2e32_v_f32m1x2(i4, vl); i4 += vl << 1;

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[2], vi0x79BD, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[7], vi1x79BD, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[12], vi2x79BD, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[17], vi3x79BD, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[22], vi4x79BD, vl);

    const vfloat32m1_t vi0xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(__riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 0), 1, vl), __riscv_vget_v_f32m1x2_f32m1(vi0xGIKMHJLN, 0), vl-1, vl);
    vi0x8ACE9BDF = vi0xGIKMHJLN;
    const vfloat32m1_t vi1xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(__riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 0), 1, vl), __riscv_vget_v_f32m1x2_f32m1(vi1xGIKMHJLN, 0), vl-1, vl);
    vi1x8ACE9BDF = vi1xGIKMHJLN;
    const vfloat32m1_t vi2xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(__riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 0), 1, vl), __riscv_vget_v_f32m1x2_f32m1(vi2xGIKMHJLN, 0), vl-1, vl);
    vi2x8ACE9BDF = vi2xGIKMHJLN;
    const vfloat32m1_t vi3xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(__riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 0), 1, vl), __riscv_vget_v_f32m1x2_f32m1(vi3xGIKMHJLN, 0), vl-1, vl);
    vi3x8ACE9BDF = vi3xGIKMHJLN;
    const vfloat32m1_t vi4xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(__riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 0), 1, vl), __riscv_vget_v_f32m1x2_f32m1(vi4xGIKMHJLN, 0), vl-1, vl);
    vi4x8ACE9BDF = vi4xGIKMHJLN;

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[5], vi0xACEG, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[10], vi1xACEG, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[15], vi2xACEG, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[20], vi3xACEG, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[25], vi4xACEG, vl);

    vo0p0 = __riscv_vfadd_vv_f32m1(vo0p0, vo0p1, vl);

    vfloat32m1_t vo0 = __riscv_vfmax_vf_f32m1(vo0p0, params->scalar.min, vl);

    vo0 = __riscv_vfmin_vf_f32m1(vo0, params->scalar.max, vl);

    __riscv_vse32_v_f32m1(o0, vo0, vl); o0 += vl;
  }
  // Last block has 1-8 pixels to process.
  assert(w <= vl << 3);
  assert(w >= 1 * sizeof(float));
  {
    vfloat32m1_t vo0p0 =  __riscv_vfmv_v_f_f32m1(weights[0], vl);

    const vfloat32m1_t vi0x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 0)), vl));
    const vfloat32m1_t vi1x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 0)), vl));
    const vfloat32m1_t vi2x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 0)), vl));
    const vfloat32m1_t vi3x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 0)), vl));
    const vfloat32m1_t vi4x8ACE = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_even, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 0)), vl));

    const vfloat32m1_t vi0x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi0x8ACE9BDF, 1)), vl));
    const vfloat32m1_t vi1x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi1x8ACE9BDF, 1)), vl));
    const vfloat32m1_t vi2x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi2x8ACE9BDF, 1)), vl));
    const vfloat32m1_t vi3x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi3x8ACE9BDF, 1)), vl));
    const vfloat32m1_t vi4x9BDF = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(vmask_odd, __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vget_v_f32m1x2_f32m1(vi4x8ACE9BDF, 1)), vl));

    vfloat32m1_t vo0p1 = __riscv_vfmul_vf_f32m1(vi0x8ACE, weights[3], vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[8], vi1x8ACE, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[13], vi2x8ACE, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[18], vi3x8ACE, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[23], vi4x8ACE, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[4], vi0x9BDF, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[9], vi1x9BDF, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[14], vi2x9BDF, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[19], vi3x9BDF, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[24], vi4x9BDF, vl);

    const vfloat32m1_t vi0x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x0246, vl-1, vl), vi0x8ACE, 1, vl);
    const vfloat32m1_t vi1x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x0246, vl-1, vl), vi1x8ACE, 1, vl);
    const vfloat32m1_t vi2x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x0246, vl-1, vl), vi2x8ACE, 1, vl);
    const vfloat32m1_t vi3x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x0246, vl-1, vl), vi3x8ACE, 1, vl);
    const vfloat32m1_t vi4x68AC = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x0246, vl-1, vl), vi4x8ACE, 1, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[1], vi0x68AC, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[6], vi1x68AC, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[11], vi2x68AC, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[16], vi3x68AC, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[21], vi4x68AC, vl);

    const vfloat32m1_t vi0x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x1357, vl-1, vl), vi0x9BDF, 1, vl);
    const vfloat32m1_t vi1x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x1357, vl-1, vl), vi1x9BDF, 1, vl);
    const vfloat32m1_t vi2x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x1357, vl-1, vl), vi2x9BDF, 1, vl);
    const vfloat32m1_t vi3x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x1357, vl-1, vl), vi3x9BDF, 1, vl);
    const vfloat32m1_t vi4x79BD = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x1357, vl-1, vl), vi4x9BDF, 1, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[2], vi0x79BD, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[7], vi1x79BD, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[12], vi2x79BD, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[17], vi3x79BD, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[22], vi4x79BD, vl);

    const vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    const vfloat32m1_t vi0xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi0x8ACE, 1, vl), vzero, vl-1, vl);
    const vfloat32m1_t vi1xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi1x8ACE, 1, vl), vzero, vl-1, vl);
    const vfloat32m1_t vi2xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi2x8ACE, 1, vl), vzero, vl-1, vl);
    const vfloat32m1_t vi3xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi3x8ACE, 1, vl), vzero, vl-1, vl);
    const vfloat32m1_t vi4xACEG = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(vi4x8ACE, 1, vl), vzero, vl-1, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[5], vi0xACEG, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[10], vi1xACEG, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[15], vi2xACEG, vl);

    vo0p1 = __riscv_vfmacc_vf_f32m1(vo0p1, weights[20], vi3xACEG, vl);

    vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, weights[25], vi4xACEG, vl);

    vo0p0 = __riscv_vfadd_vv_f32m1(vo0p0, vo0p1, vl);

    vfloat32m1_t vo0 = __riscv_vfmax_vf_f32m1(vo0p0, params->scalar.min, vl);

    vo0 = __riscv_vfmin_vf_f32m1(vo0, params->scalar.max, vl);

    size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
    __riscv_vse32_v_f32m1(o0, vo0, w_tmp); o0 += w_tmp;
  }

  i0 = (const float*) ((uintptr_t) i2 - input_decrement);
  i1 = (const float*) ((uintptr_t) i3 - input_decrement);
  i2 = (const float*) ((uintptr_t) i4 - input_decrement);
  i3 = (const float*) ((uintptr_t) i2 + input_width);
  i4 = (const float*) ((uintptr_t) i3 + input_width);


  output_height -= 1;
  padded_input_height -= 2;
} while (output_height != 0);
}
