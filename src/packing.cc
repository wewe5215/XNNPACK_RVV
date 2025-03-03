// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/config-types.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pack.h"
#include "xnnpack/unaligned.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qsu4cxs1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0.h"
#endif  // XNN_ENABLE_KLEIDIAI

#include <fp16/fp16.h>

extern "C" {

void xnn_pack_f32_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = 0;
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (float*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
#include <riscv_vector.h>
void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x1v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
      int output_padding_down_stride = output_padding_top_stride;
      while(output_padding_top_stride != 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          int padded_k_h = input_padding_top - (out_h << 1);
          in_ptr = input + input_cursor;
          out_ptr += vlmax * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += (output_width << 1);
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      input_cursor = base + input_width;
      while(((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - 2)) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - 2)) + (out_w << 1);
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          out_ptr += k_h_padding_end * vlmax * group_input_channels * kernel_width;
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      output_cur = 0;
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x1v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  batch = im2col_cur / output_size;
  // top_pad
  /*
      output_cur < vlmax * ceil(output_width / vlmax)
  --> output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)
  */ 
  int input_cursor_real_part = 0;
  for(batch = 0; batch < batch_size; batch ++){
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_whole_stride_padded_part = -(output_padding_top_stride/vlmax >= 1);
          remainder = ~is_whole_stride_padded_part & output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          uint32_t* in_ptr_now_real_part = in_ptr + output_size*batch + (-(remainder == 0) & input_cursor_real_part);
          // std::cout << "first: output_cur = " << output_cur << ", remainder = " << remainder << "\n";
          out_ptr += (is_whole_stride_padded_part & (min(vlmax, output_width) * group_input_channels * kernel_width));
          output_padding_top_stride -= (is_whole_stride_padded_part & vlmax);
          for(int k_h = is_whole_stride_padded_part & input_padding_top; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int moved_input_ptr_real_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w) & (-((output_cur + remainder) % output_width == 0));
                  int output_padding_right = zero_max(k_w + output_width-valid_width) & (-((output_cur + vlmax) % output_width == 0));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      // |---- pad_top ---- |  --> output_padding_top_stride 
                      // |- real -| -pad_r -|  --> remainder_padding_right(for k_w == 2 when kernel_width = 3)
                      out_ptr += output_padding_left & -(remainder > 0 && out_w == 0);
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      // std::cout << "output_padding_left = " << output_padding_left << ", vl = " << vl << "\n";
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      out_ptr += output_padding_left;
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m1(in_ptr_now_real_part, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + output_padding_right;
                      in_ptr_now += input_size;
                      in_ptr_now_real_part += input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  in_ptr_now_real_part = in_ptr_now_real_part - input_size*group_input_channels + 1 - (output_padding_left & (-((output_cur + remainder) % output_width == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
                  moved_input_ptr_real_step += (1 - (output_padding_left & (-((output_cur + remainder) % output_width == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              in_ptr_now_real_part = in_ptr_now_real_part - moved_input_ptr_real_step + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          // todo : input_offset, end height padding
          int input_offset = -(output_padding_top_stride == 0 && output_width % vlmax != 0) & (vlmax - remainder + (-(vlmax - remainder < output_width) & -input_padding_left));
          input_cursor = input_cursor + input_offset;
          input_cursor_real_part += (vlmax - remainder - (-(output_cur == 0) & input_padding_left));
          im2col_cur += vlmax;
          output_cur += vlmax;
          // std::cout << "first: input_offset= " << input_offset << "\n";
      }
      // middle part
      while((output_cur + (-((output_cur + vlmax) % output_width) & vlmax)) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          // -(out_h < input_padding_top) --> for the condition with padding
          // 0xFFFFFFFF if cond=1, else 0x00000000
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "middle: output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          // || -- pad -- | -- input ----- ||                         --> pad before second load/store
          // || -- remainder ------------- || -- pad -- | - input -|| --> pad before second load/store
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          // || --- input ---   |-- pad -- ||                         --> pad after second load/store
          // || -- remainder -- |-- pad -- || ------ input ------- || --> pad before second load/store
          int is_in_right_part1 = -((out_h != (output_cur + vlmax) / output_width));
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (is_in_left_part & zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = (-(remainder == 0 || (remainder > 0 && ~is_in_right_part2)) & is_in_right_part1 & zero_max(k_w + output_width-valid_width));
                  int vl;
                  // std::cout << "remainder_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      // std::cout << "vl = " << vl << "\n";
                      v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);

                      out_ptr += vl + output_padding_right;
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          // || -- pad -- | -- input ----- ||
          //      --> + len(input)
          // || -- remainder ------------- || -- pad -- | - input -|| 
          //      --> + remainder - (output_width - input_padding_left) + input_width + (vlmax - remainder - input_padding_left) \
                      = remainder - output_width + input_padding_left + input_width + vlmax - remainder - input_padding_left \
                      = vlmax
          // || ----------- input -------- ||
          //      --> + vlmax
          // || --- pad( == remainder) --- || -- pad -- | - input -||
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (vlmax - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          // std::cout << "middle: input_offset= " << input_offset << "\n";
          // std::cout << "middle: input_cursor= " << input_cursor << "\n";
          im2col_cur += vlmax;
          output_cur += vlmax;
      }

      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          // std::cout << "is_next_batch = " << (is_next_batch & 1) << "\n";
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "end: output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -(vlmax - remainder >= output_width || (remainder == 0 && (out_h != (output_cur + vlmax) / output_width))); 
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = is_in_left_part & zero_max(input_padding_left-k_w);
                  int output_padding_right = is_in_right_part1 & zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = zero_max(k_w + output_width-valid_width) & (-(remainder > 0 && out_h != (output_cur + vlmax) / output_width));
                  int vl;
                  // std::cout << "k_h = " << k_h << ", k_w = " << k_w << ", output_padding_left = " << output_padding_left << "\n";
                  // std::cout << "output_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -(
                          (zero_max(k_h - input_padding_top)) && out_h >= output_height - input_padding_top
                      );
                      vl = zero_max(remainder - remainder_padding_right);
                      out_ptr += remainder_padding_left;
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - output_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int is_padded = -(
                                      ((zero_max(k_h - input_padding_top)) & -(((output_cur + vlmax) / output_width + 1 > output_height - input_padding_top))) \
                                      || ((output_cur + remainder >= output_size) && (batch + 1 >= batch_size)) \
                                      || (zero_max(input_padding_top-k_h) & -(output_cur + remainder >= output_size && batch + 1 != batch_size))
                                      );
                      int is_in_next_batch = -(output_cur + remainder >= output_size && batch + 1 != batch_size);
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded | is_padded_remainder) & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded | is_padded_remainder) & vl);

                      out_ptr += (vl + output_padding_right);
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (vlmax - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += vlmax;
          output_cur += vlmax;
          // std::cout << "end: input_offset = " << input_offset << "\n";

      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      // std::cout << "end: finished_part_in_next_batch = " << finished_part_in_next_batch << "\n";
      input_cursor = output_size*(batch + 1);// + ((vlmax - remainder) % output_width);// output_size = output_width*output_height = input_width*input_height
      input_cursor_real_part = finished_part_in_next_batch - (finished_part_in_next_batch + output_width - 1) / output_width;
      // std::cout << "output_padding_top_stride = " << output_padding_top_stride << ", input_cursor = " << input_cursor << "\n";
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x2v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
      int output_padding_down_stride = output_padding_top_stride;
      while(output_padding_top_stride != 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          int padded_k_h = input_padding_top - (out_h << 1);
          in_ptr = input + input_cursor;
          out_ptr += vlmax * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += (output_width << 1);
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      input_cursor = base + input_width;
      while(((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - 2)) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - 2)) + (out_w << 1);
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          out_ptr += k_h_padding_end * vlmax * group_input_channels * kernel_width;
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      output_cur = 0;
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x2v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m2();
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              (cond & nr));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x2v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  batch = im2col_cur / output_size;
  // top_pad
  /*
      output_cur < vlmax * ceil(output_width / vlmax)
  --> output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)
  */ 
  int input_cursor_real_part = 0;
  for(batch = 0; batch < batch_size; batch ++){
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_whole_stride_padded_part = -(output_padding_top_stride/vlmax >= 1);
          remainder = ~is_whole_stride_padded_part & output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          uint32_t* in_ptr_now_real_part = in_ptr + output_size*batch + (-(remainder == 0) & input_cursor_real_part);
          // std::cout << "first: output_cur = " << output_cur << ", remainder = " << remainder << "\n";
          out_ptr += (is_whole_stride_padded_part & (min(vlmax, output_width) * group_input_channels * kernel_width));
          output_padding_top_stride -= (is_whole_stride_padded_part & vlmax);
          for(int k_h = is_whole_stride_padded_part & input_padding_top; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int moved_input_ptr_real_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w) & (-((output_cur + remainder) % output_width == 0));
                  int output_padding_right = zero_max(k_w + output_width-valid_width) & (-((output_cur + vlmax) % output_width == 0));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += output_padding_left & -(remainder > 0 && out_w == 0);
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      // std::cout << "output_padding_left = " << output_padding_left << ", vl = " << vl << "\n";
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      out_ptr += output_padding_left;
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m2(in_ptr_now_real_part, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + output_padding_right;
                      in_ptr_now += input_size;
                      in_ptr_now_real_part += input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  in_ptr_now_real_part = in_ptr_now_real_part - input_size*group_input_channels + 1 - (output_padding_left & (-((output_cur + remainder) % output_width == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
                  moved_input_ptr_real_step += (1 - (output_padding_left & (-((output_cur + remainder) % output_width == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              in_ptr_now_real_part = in_ptr_now_real_part - moved_input_ptr_real_step + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          // todo : input_offset, end height padding
          int input_offset = -(output_padding_top_stride == 0 && output_width % vlmax != 0) & (vlmax - remainder + (-(vlmax - remainder < output_width) & -input_padding_left));
          input_cursor = input_cursor + input_offset;
          input_cursor_real_part += (vlmax - remainder - (-(output_cur == 0) & input_padding_left));
          im2col_cur += vlmax;
          output_cur += vlmax;
          // std::cout << "first: input_offset= " << input_offset << "\n";
      }
      // middle part
      while(((output_cur + vlmax) % output_width != 0 && (output_cur + vlmax) / output_width < output_height - input_padding_top) || \
            ((output_cur + vlmax) % output_width == 0 && (output_cur + vlmax) / output_width <= output_height - input_padding_top)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          // -(out_h < input_padding_top) --> for the condition with padding
          // 0xFFFFFFFF if cond=1, else 0x00000000
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "middle: output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          // || -- pad -- | -- input ----- ||                         --> pad before second load/store
          // || -- remainder ------------- || -- pad -- | - input -|| --> pad before second load/store
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          // || --- input ---   |-- pad -- ||                         --> pad after second load/store
          // || -- remainder -- |-- pad -- || ------ input ------- || --> pad before second load/store
          int is_in_right_part1 = -((out_h != (output_cur + vlmax) / output_width));
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (is_in_left_part & zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = (-(remainder == 0 || (remainder > 0 && ~is_in_right_part2)) & is_in_right_part1 & zero_max(k_w + output_width-valid_width));
                  int vl;
                  // std::cout << "remainder_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      // std::cout << "vl = " << vl << "\n";
                      v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);

                      out_ptr += vl + output_padding_right;
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          // || -- pad -- | -- input ----- ||
          //      --> + len(input)
          // || -- remainder ------------- || -- pad -- | - input -|| 
          //      --> + remainder - (output_width - input_padding_left) + input_width + (vlmax - remainder - input_padding_left) \
                      = remainder - output_width + input_padding_left + input_width + vlmax - remainder - input_padding_left \
                      = vlmax
          // || ----------- input -------- ||
          //      --> + vlmax
          // || --- pad( == remainder) --- || -- pad -- | - input -||
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (vlmax - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          // std::cout << "middle: input_offset= " << input_offset << "\n";
          // std::cout << "middle: input_cursor= " << input_cursor << "\n";
          im2col_cur += vlmax;
          output_cur += vlmax;
      }

      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          // std::cout << "is_next_batch = " << (is_next_batch & 1) << "\n";
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "end: output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -(vlmax - remainder >= output_width || (remainder == 0 && (out_h != (output_cur + vlmax) / output_width))); 
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = is_in_left_part & zero_max(input_padding_left-k_w);
                  int output_padding_right = is_in_right_part1 & zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = zero_max(k_w + output_width-valid_width) & (-(remainder > 0 && out_h != (output_cur + vlmax) / output_width));
                  int vl;
                  // std::cout << "k_h = " << k_h << ", k_w = " << k_w << ", output_padding_left = " << output_padding_left << "\n";
                  // std::cout << "output_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -(
                          (zero_max(k_h - input_padding_top)) && out_h >= output_height - input_padding_top
                      );
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      out_ptr += remainder_padding_left;
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - output_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int is_padded = -(
                                      ((zero_max(k_h - input_padding_top)) & -(((output_cur + vlmax) / output_width + 1 > output_height - input_padding_top))) \
                                      || ((output_cur + remainder >= output_size) && (batch + 1 >= batch_size)) \
                                      || (zero_max(input_padding_top-k_h) & -(output_cur + remainder >= output_size && batch + 1 != batch_size))
                                      );
                      int is_in_next_batch = -(output_cur + remainder >= output_size && batch + 1 != batch_size);
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, (~is_padded | is_padded_remainder) & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, (~is_padded | is_padded_remainder) & vl);

                      out_ptr += (vl + output_padding_right);
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (vlmax - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += vlmax;
          output_cur += vlmax;
          // std::cout << "end: input_offset = " << input_offset << "\n";

      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      // std::cout << "end: finished_part_in_next_batch = " << finished_part_in_next_batch << "\n";
      input_cursor = output_size*(batch + 1);// + ((vlmax - remainder) % output_width);// output_size = output_width*output_height = input_width*input_height
      input_cursor_real_part = finished_part_in_next_batch - (finished_part_in_next_batch + output_width - 1) / output_width;
      // std::cout << "output_padding_top_stride = " << output_padding_top_stride << ", input_cursor = " << input_cursor << "\n";
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
      int output_padding_down_stride = output_padding_top_stride;
      while(output_padding_top_stride != 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          int remainder = -(output_width - out_w < vlmax) & (output_width - out_w);
          // std::cout << "remainder = " << remainder << "\n";
          // replace `* stride` to `<< 1` by the fact that stride = 2
          int padded_k_h = input_padding_top - (out_h << 1) - (-(remainder > 0) & ((out_h + 1) << 1));
          in_ptr = input + input_cursor;
          in_ptr_rem = input + base;
          out_ptr += vlmax * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && k_h < input_padding_top - (out_h << 1));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m4_t v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_remainder, stride_width << 2, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += ~padded & (output_width << 1);
              in_ptr_rem += -(remainder > 0) & (output_width << 1);
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      input_cursor = base + input_width;
      while(((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          in_ptr_rem = input + input_cursor_rem;
          int remainder = -(output_width - out_w < vlmax) & (output_width - out_w);
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && ((input_padding_top + 1) >> 1) + input_cursor_rem / input_width + k_h - base / input_width >= valid_height);
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m4_t v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_remainder, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += output_width << 1;
              in_ptr_rem += -(remainder > 0) & (output_width << 1);
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - 2)) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - 2)) + (out_w << 1);
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          int remainder = -(output_width - out_w < vlmax) & (output_width - out_w);
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m4_t v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          out_ptr += k_h_padding_end * vlmax * group_input_channels * kernel_width;
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      output_cur = 0;
  }
}


void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m4();
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_cur < output_width * ((output_width + nr - 1) / nr)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              (cond & nr));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}


XNN_INTERNAL void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m4();
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_cur < output_width * ((output_width + nr - 1) / nr)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (nr - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          // std::cout << "input_cursor = " << input_cursor << "\n";
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}


XNN_INTERNAL void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  batch = im2col_cur / output_size;
  // top_pad
  /*
      output_cur < vlmax * ceil(output_width / vlmax)
  --> output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)
  */ 
  for(batch = 0; batch < batch_size; batch ++){
      while(output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_whole_stride_padded_part = -(output_padding_top_stride/vlmax >= 1);
          remainder = ~is_whole_stride_padded_part & output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          out_ptr += (is_whole_stride_padded_part & (min(vlmax, output_width) * group_input_channels * kernel_width));
          output_padding_top_stride -= (is_whole_stride_padded_part & vlmax);
          for(int k_h = is_whole_stride_padded_part & input_padding_top; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      // |---- pad_top ---- |  --> output_padding_top_stride 
                      // |- real -| -pad_r -|  --> remainder_padding_right(for k_w == 2 when kernel_width = 3)
                      out_ptr += output_padding_left & -(remainder > 0 && out_w == 0);
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      // std::cout << "output_padding_left = " << output_padding_left << ", vl = " << vl << "\n";
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      out_ptr += output_padding_left;
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "is_added = " << is_added << "\n";
                      vl = vlmax - remainder - output_padding_left;
                      v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += vl;
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
                  // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          // todo : input_offset, end height padding
          int input_offset = -(output_padding_top_stride == 0) & (vlmax - remainder - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += vlmax;
          output_cur += vlmax;
      }
      // middle part
      while((output_cur + vlmax) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          // -(out_h < input_padding_top) --> for the condition with padding
          // 0xFFFFFFFF if cond=1, else 0x00000000
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          // || -- pad -- | -- input ----- ||                         --> pad before second load/store
          // || -- remainder ------------- || -- pad -- | - input -|| --> pad before second load/store
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          // || --- input ---   |-- pad -- ||                         --> pad after second load/store
          // || -- remainder -- |-- pad -- || ------ input ------- || --> pad before second load/store
          int is_in_right_part1 = -((out_h != (output_cur + vlmax) / output_width));
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (is_in_left_part & zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = (-(remainder == 0 || (remainder > 0 && ~is_in_right_part2)) & is_in_right_part1 & zero_max(k_w + output_width-valid_width));
                  int vl;
                  // std::cout << "remainder_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      // std::cout << "vl = " << vl << "\n";
                      v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);

                      out_ptr += vl + output_padding_right;
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          // || -- pad -- | -- input ----- ||
          //      --> + len(input)
          // || -- remainder ------------- || -- pad -- | - input -|| 
          //      --> + remainder - (output_width - input_padding_left) + input_width + (vlmax - remainder - input_padding_left) \
                      = remainder - output_width + input_padding_left + input_width + vlmax - remainder - input_padding_left \
                      = vlmax
          // || ----------- input -------- ||
          //      --> + vlmax
          // || --- pad( == remainder) --- || -- pad -- | - input -||
          int cond = -(input_cursor % input_width == 0);
          // todo : input_offset, end height padding
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (vlmax - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          // std::cout << "input_offset = " << input_offset << "\n";
          im2col_cur += vlmax;
          output_cur += vlmax;
      }

      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          // std::cout << "is_next_batch = " << (is_next_batch & 1) << "\n";
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -(vlmax - remainder >= output_width || remainder == 0); 
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = is_in_left_part & zero_max(input_padding_left-k_w);
                  int output_padding_right = is_in_right_part1 & zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = zero_max(k_w + output_width-valid_width) & (-(remainder > 0 && out_h != (output_cur + vlmax) / output_width));
                  int vl;
                  // std::cout << "k_h = " << k_h << ", k_w = " << k_w << ", output_padding_left = " << output_padding_left << "\n";
                  // std::cout << "output_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -(
                          (zero_max(k_h - input_padding_top)) && out_h >= output_height - input_padding_top
                      );
                      vl = zero_max(remainder - remainder_padding_right);
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - output_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int is_padded = -(
                                      ((zero_max(k_h - input_padding_top)) & -(((output_cur + vlmax) / output_width + 1 > output_height - input_padding_top))) \
                                      || ((output_cur + remainder >= output_size) && (batch + 1 >= batch_size)) \
                                      || (zero_max(input_padding_top-k_h) & -(output_cur + remainder >= output_size && batch + 1 != batch_size))
                                      );
                      int is_in_next_batch = -(output_cur + remainder >= output_size && batch + 1 != batch_size);
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, (~is_padded | is_padded_remainder) & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, (~is_padded | is_padded_remainder) & vl);

                      out_ptr += (vl + output_padding_right);
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (vlmax - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += vlmax;
          output_cur += vlmax;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      input_cursor = output_size*(batch + 1);// + ((vlmax - remainder) % output_width);// output_size = output_width*output_height = input_width*input_height
      // std::cout << "output_padding_top_stride = " << output_padding_top_stride << ", input_cursor = " << input_cursor << "\n";
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  batch = im2col_cur / output_size;
  // top_pad
  /*
      output_cur < vlmax * ceil(output_width / vlmax)
  --> output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)
  */ 
  for(batch = 0; batch < batch_size; batch ++){
      while(output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_whole_stride_padded_part = -(output_padding_top_stride/vlmax >= 1);
          remainder = ~is_whole_stride_padded_part & output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          out_ptr += (is_whole_stride_padded_part & (min(vlmax, output_width) * group_input_channels * kernel_width));
          output_padding_top_stride -= (is_whole_stride_padded_part & vlmax);
          for(int k_h = is_whole_stride_padded_part & input_padding_top; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += output_padding_left & -(remainder > 0 && out_w == 0);
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      memcpy(out_ptr, in_ptr_now, vl << 2);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      out_ptr += output_padding_left;
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      vl = vlmax - remainder - output_padding_left;
                      memcpy(out_ptr, in_ptr_now, vl << 2);
                      out_ptr += vl;
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          // todo : input_offset, end height padding
          int input_offset = -(output_padding_top_stride == 0) & (vlmax - remainder - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += vlmax;
          output_cur += vlmax;
      }
      // middle part
      while((output_cur + vlmax) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          // -(out_h < input_padding_top) --> for the condition with padding
          // 0xFFFFFFFF if cond=1, else 0x00000000
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + vlmax) / output_width));
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (is_in_left_part & zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = (-(remainder == 0 || (remainder > 0 && ~is_in_right_part2)) & is_in_right_part1 & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      memcpy(out_ptr, in_ptr_now, vl << 2);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      memcpy(out_ptr, in_ptr_now, vl << 2);

                      out_ptr += vl + output_padding_right;
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (vlmax - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += vlmax;
          output_cur += vlmax;
      }

      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -(vlmax - remainder >= output_width || remainder == 0); 
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = is_in_left_part & zero_max(input_padding_left-k_w);
                  int output_padding_right = is_in_right_part1 & zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = zero_max(k_w + output_width-valid_width) & (-(remainder > 0 && out_h != (output_cur + vlmax) / output_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -(
                          (zero_max(k_h - input_padding_top)) && out_h >= output_height - input_padding_top
                      );
                      vl = zero_max(remainder - remainder_padding_right);
                      memcpy(out_ptr, in_ptr_now, (~is_padded_remainder & vl) << 2);
                      // remainder with padding
                      out_ptr = out_ptr + remainder + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - output_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int is_padded = -(
                                      ((zero_max(k_h - input_padding_top)) & -(((output_cur + vlmax) / output_width + 1 > output_height - input_padding_top))) \
                                      || ((output_cur + remainder >= output_size) && (batch + 1 >= batch_size)) \
                                      || (zero_max(input_padding_top-k_h) & -(output_cur + remainder >= output_size && batch + 1 != batch_size))
                                      );
                      int is_in_next_batch = -(output_cur + remainder >= output_size && batch + 1 != batch_size);
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      memcpy(out_ptr, in_ptr_now, ((~is_padded | is_padded_remainder) & vl) << 2);
                      out_ptr += (vl + output_padding_right);
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (vlmax - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += vlmax;
          output_cur += vlmax;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      input_cursor = output_size*(batch + 1);// + ((vlmax - remainder) % output_width);// output_size = output_width*output_height = input_width*input_height
      // std::cout << "output_padding_top_stride = " << output_padding_top_stride << ", input_cursor = " << input_cursor << "\n";
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m8();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  int remainder = 0;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
      int output_padding_down_stride = output_padding_top_stride;
      while(output_padding_top_stride > 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          // replace `* stride` to `<< 1` by the fact that stride = 2
          int padded_k_h = zero_max(input_padding_top - (out_h << 1) - (-(remainder > 0) & ((out_h + 1) << 1)));
          // std::cout << "remainder = " << remainder << ", padded_k_h = " << padded_k_h << "\n";
          in_ptr = input + input_cursor;
          in_ptr_rem = input + base + ((output_cur + remainder) / output_width) * output_width;
          out_ptr += vlmax * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && k_h < input_padding_top - (out_h << 1));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m8_t v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_now, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_remainder, stride_width << 2, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += ~padded & (output_width << 1);
              in_ptr_rem += -(remainder > 0) & (output_width << 1);
              // std::cout << "moved offset = " << (~padded & (output_width << 1)) << "\n";
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      input_cursor = base + input_width + (remainder << 1);
      while(((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          in_ptr_rem = input + input_cursor_rem;
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && ((input_padding_top + 1) >> 1) + input_cursor_rem / input_width + k_h - base / input_width >= valid_height);
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m8_t v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_remainder, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += output_width << 1;
              in_ptr_rem += output_width << 1;
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - 2)) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - 2)) + (out_w << 1);
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m8_t v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          out_ptr += k_h_padding_end * vlmax * group_input_channels * kernel_width;
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      output_cur = 0;
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m8();
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int current_output_offset = output_cur + remainder;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(im2col_cur + remainder + cur_vl >= input_size);
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          current_output_offset = current_output_offset + output_width - (is_exceed_boarder & output_size);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              (cond & nr));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m8();
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_cur < output_width * ((output_width + nr - 1) / nr)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (nr - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          // std::cout << "input_cursor = " << input_cursor << "\n";
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m8();
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                          __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                          __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m4(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (nr - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          // std::cout << "input_cursor = " << input_cursor << "\n";
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      input_cursor = output_size * (batch + 1) + (-(output_padding_top_stride == 0) & ((output_cur / output_width - 1) * output_width + (output_cur % output_width - input_padding_left)));
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_8x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m8();
  const int vlmax = __riscv_vsetvlmax_e32m8();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m8_t v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, vl);
                          __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m8_t v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, vl);
                          __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m8_t v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m8(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (nr - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          // std::cout << "input_cursor = " << input_cursor << "\n";
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      input_cursor = output_size * (batch + 1) + (-(output_padding_top_stride == 0) & ((output_cur / output_width - 1) * output_width + (output_cur % output_width - input_padding_left)));
  }
}

#endif

void xnn_pack_f16_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = 0;
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = xnn_float16_from_float(b[nr_block_start + nr_block_offset]);
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = xnn_float16_zero();
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = xnn_float16_from_float(k[(nr_block_start + nr_block_offset) * kc + kc_idx]);
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (xnn_float16*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t bzp = (int32_t) kc * izp * (int32_t) params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, bzp);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const uint8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ksum += (int32_t) kv;
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (uint8_t*) packed_weights + kr;
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ksum += (uint32_t) kv;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ksum += (uint32_t) kv;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

static int8_t sign_extend_int4(int8_t value) {
  return (value ^ 0x8) - 8;
}

void xnn_pack_qs8_qc4w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t) params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = (nr_block_start + nr_block_offset) * kc + kc_idx;
            const size_t kh_offset = k_offset + kr;
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ksum += kv_lo + kv_hi - 2 * kernel_zero_point;  // subtract 2 zero points
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp * 16);
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

// Same as qc4w but unsigned 4 bit output
// Applies kv ^ 0x88 to convert int4 to uint4
void xnn_pack_qs8_qc4uw_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t) params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = (nr_block_start + nr_block_offset) * kc + kc_idx;
            const size_t kh_offset = k_offset + kr;
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*) packed_weights)[kr_block_offset] = kv ^ 0x88; // Convert to uint4
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ksum += kv_lo + kv_hi - 2 * kernel_zero_point;  // subtract 2 zero points
              ((uint8_t*) packed_weights)[kr_block_offset] = kv ^ 0x88; // Convert to uint4
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp * 16);
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_qb4w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t bl,             // blocksize
  const uint8_t* k,      // kernel
  const float* bias,
  const xnn_bfloat16* scale,
  void* packed_weights,
  size_t extra_bytes_bl, // extra bytes per block
  size_t extra_bytes_n,  // extra bytes per n
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8);
  assert(bias == nullptr); // Not used here. Must be updated outside.

  const size_t skr = sr * kr;

  // Constraints for blocksize
  // These need to be reevaluated in the future.
  assert(bl != 0);
  assert(round_up_po2(kc, skr) % bl == 0); // must be round number of blocks inside a column
  assert(bl % skr == 0); // must be round number of kr * sr
  assert(bl <= round_up_po2(kc, skr)); // must not be larger than K
  assert(2 * skr <= bl); // must be at least two skr to avoid back-to-back extra_bytes

  const size_t num_blocks = round_up_po2(kc, skr) / bl;
  const int32_t izp = (int32_t) params->input_zero_point;

  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      float* packed_b = (float*) packed_weights;
      packed_weights = (float*) packed_weights + nr_block_size;
      packed_weights = (float*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = (nr_block_start + nr_block_offset) * kc + kc_idx;
            const size_t kh_offset = k_offset + kr;
            uint8_t kv_lo = 8;
            if (kc_idx < kc) {
              kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
            }
            uint8_t kv_hi = 8;
            if ((kc_idx + kr) < kc) {
              kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
            }
            ksum += kv_lo + kv_hi - 16;  // subtract 2 zero points (8)
            const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
            ((uint8_t*) packed_weights)[kr_block_offset] = kv;
          }

          size_t block_index = kr_block_start / bl;
          size_t scale_index = (nr_block_start + nr_block_offset) * num_blocks + block_index;
          unaligned_indexed_store_f32(packed_b, nr_block_offset,
            unaligned_indexed_load_f32(packed_b, nr_block_offset) -
              (float) ksum * izp * xnn_bfloat16_to_float(scale[scale_index]));
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        if (((2 * kr) + kr_block_start) % bl == 0) {
          packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes_bl);
        }

        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes_n);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
  } while (--g != 0);
}

void xnn_pack_qs8_qb4w_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  size_t bl,              // block size
  const uint8_t* k,       // kernel
  const float* bias,
  const xnn_bfloat16* scale,  // block scales (bf16 format)
  void* packed_weights,
  size_t extra_bytes_bl,  // extra bytes per block
  size_t extra_bytes_n,   // extra bytes per n
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8);
  assert(bias == nullptr);  // Not used here. Must be updated outside.

  const size_t skr = sr * kr;

  // Constraints for blocksize
  // These need to be reevaluated in the future.
  assert(bl != 0);
  assert(round_up_po2(kc, skr) % bl == 0); // must be round number of blocks inside a column
  assert(bl % skr == 0); // must be round number of kr * sr
  assert(bl <= round_up_po2(kc, skr)); // must not be larger than K
  assert(2 * skr <= bl); // must be at least two skr to avoid back-to-back extra_bytes

  const size_t num_blocks = round_up_po2(kc, skr) / bl;
  const int32_t izp = (int32_t) params->input_zero_point;

  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      packed_weights = (float*) packed_weights + nr_block_size;
      packed_weights = (float*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = (nr_block_start + nr_block_offset + kc_idx * k_stride);
            const size_t kh_offset = k_offset + (kr * k_stride);
            uint8_t kv_lo = 8;
            if (kc_idx < kc) {
              kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
            }
            uint8_t kv_hi = 8;
            if ((kc_idx + kr) < kc) {
              kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
            }
            ksum += kv_lo + kv_hi - 16;  // subtract 2 zero points (8)
            const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
            ((uint8_t*) packed_weights)[kr_block_offset] = kv;
          }

          size_t block_index = kr_block_start / bl;
          size_t scale_index = (nr_block_start + nr_block_offset) * num_blocks + block_index;
          unaligned_indexed_store_f32(packed_b, nr_block_offset,
            unaligned_indexed_load_f32(packed_b, nr_block_offset) -
              (float) ksum * izp * xnn_bfloat16_to_float(scale[scale_index]));
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        if (((2 * kr) + kr_block_start) % bl == 0) {
          packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes_bl);
        }

        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes_n);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
  } while (--g != 0);
}

void xnn_pack_qs8_qc4w_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t) params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = kc_idx * k_stride + (nr_block_start + nr_block_offset);
            const size_t kh_offset = (kc_idx + kr) * k_stride + (nr_block_start + nr_block_offset);
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              ksum += kv_lo + kv_hi - 2 * kernel_zero_point;  // subtract 2 zero points
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp * 16);
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

// Same as qc4w but unsigned 4 bit output
// Applies kv ^ 0x88 to convert int4 to uint4
void xnn_pack_qs8_qc4uw_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t) params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = kc_idx * k_stride + (nr_block_start + nr_block_offset);
            const size_t kh_offset = (kc_idx + kr) * k_stride + (nr_block_start + nr_block_offset);
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*) packed_weights)[kr_block_offset] = kv ^ 0x88; // Convert to uint4
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              ksum += kv_lo + kv_hi - 2 * kernel_zero_point;  // subtract 2 zero points
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ((uint8_t*) packed_weights)[kr_block_offset] = kv ^ 0x88; // Convert to uint4
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp * 16);
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_qs8w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const int32_t* b = (const int32_t*) bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

// qs4 packs 2 columns into 2 rows.
// kc can be odd.  assume k values in a row are padded to a byte boundary
void xnn_pack_f32_qc4w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* k,  // 4 bit values
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  kc = (kc + 1) >> 1;
  const int32_t* b = (const int32_t*) bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const uint8_t kv = ((const uint8_t*) k)[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_weights = (uint8_t*) packed_weights + kr;
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k = (const uint8_t*) k + nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = 0.0f;
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[kc_idx * k_stride + nr_block_start + nr_block_offset];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (float*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f16_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = UINT16_C(0);
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[kc_idx * k_stride + nr_block_start + nr_block_offset];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const float* k,
  const float* b,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = xnn_float16_from_float(b[nr_block_start + nr_block_offset]);
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = xnn_float16_zero();
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = xnn_float16_from_float(k[kc_idx * k_stride + nr_block_start + nr_block_offset]);
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (xnn_float16*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t bzp = (int32_t) kc * izp * (int32_t) params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, bzp);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const uint8_t kv = k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ksum += (int32_t) kv;
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (uint8_t*) packed_weights + kr;
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (uint32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ksum += (uint32_t) kv;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (uint32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ksum += (uint32_t) kv;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void pack_weights_and_biases(uint32_t flags,                                 //
                             const struct xnn_gemm_config* gemm_config,      //
                             size_t input_channels,                          //
                             size_t output_channels,                         //
                             size_t groups,                                  //
                             size_t weights_stride,                          //
                             xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio_w,  //
                             xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w,  //
                             const void* accumulator_init,                   //
                             const void* weights,                            //
                             xnn_init_scale_params_fn init_extra_data0_fn,   //
                             const void* extra_data0,                        //
                             size_t extra_data0_element_size,                //
                             xnn_init_scale_params_fn init_extra_data1_fn,   //
                             const void* extra_data1,                        //
                             size_t extra_data1_element_size,                //
                             void* packed_weights_ptr,                       //
                             size_t extra_bytes,                             //
                             const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t n_stride = round_up(output_channels, nr);
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    pack_gemm_gio_w(
      groups, output_channels, input_channels,
      nr, kr, sr,
      output_channels,
      weights, accumulator_init, /*scale=*/nullptr,
      packed_weights_ptr,
      nr * extra_bytes,
      params);
  } else {
    pack_gemm_goi_w(
      groups, output_channels, input_channels,
      nr, kr, sr,
      weights, accumulator_init, /*scale=*/nullptr,
      packed_weights_ptr,
      nr * extra_bytes,
      params);
  }
  if (extra_data1 != nullptr) {
    assert(init_extra_data1_fn != nullptr);

    for (size_t group = 0; group < groups; group++) {
      void* packed_group_ptr = (void*)((char*)packed_weights_ptr +
                                       group * n_stride * weights_stride);
      void* weights = (void*)((uintptr_t)packed_group_ptr +
                              nr * (weights_stride - extra_bytes));
      void* extra_data_ptr =
          (void*)((uintptr_t)extra_data1 +
                  extra_data1_element_size * output_channels * group);
      init_extra_data1_fn(output_channels, nr, nr, nr * weights_stride,
                          nr * weights_stride, 0, extra_data_ptr, weights);
    }
  }

  if (extra_data0 != nullptr) {
    assert(init_extra_data0_fn != nullptr);
    for (size_t group = 0; group < groups; group++) {
      void* packed_group_ptr = (void*)((char*)packed_weights_ptr +
                                       group * n_stride * weights_stride);
      void* weights = (void*)((uintptr_t)packed_group_ptr +
                              nr * (weights_stride - extra_bytes));
      if (extra_data1 != nullptr) {
        weights = (void*)((uintptr_t)weights + nr * sizeof(float));
      }
      void* extra_data_ptr =
          (void*)((uintptr_t)extra_data0 +
                  extra_data0_element_size * output_channels * group);
      init_extra_data0_fn(output_channels, nr, nr, nr * weights_stride,
                          nr * weights_stride, 0, extra_data_ptr, weights);
    }
  }
}

size_t xnn_packed_stride_qs8_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t unused_k, size_t k_stride,
    size_t extra_bytes) {
  const size_t bias_element_size = sizeof(int32_t);
  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_INT8_T;
  return (k_stride << log2_filter_element_size) + bias_element_size +
         extra_bytes;
}

void xnn_pack_qs8_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const size_t extra_bytes =
      extra_data0_element_size + extra_data1_element_size;
  const size_t weights_stride = xnn_packed_stride_qs8_weights_and_biases(
      gemm_config, input_channels, k_stride, extra_bytes);
  return pack_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      weights_stride, (xnn_packw_gemm_gio_ukernel_fn)xnn_pack_qs8_gemm_gio_w,
      (xnn_packw_gemm_goi_ukernel_fn)xnn_pack_qs8_gemm_goi_w, accumulator_init,
      weights, init_extra_data0_fn, extra_data0, extra_data0_element_size,
      init_extra_data1_fn, extra_data1, extra_data1_element_size,
      packed_weights_ptr, extra_bytes, params);
}

size_t xnn_packed_stride_qs4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t unused_k, size_t k_stride,
    size_t extra_bytes) {
  const size_t bias_element_size = sizeof(int32_t);
  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_INT8_T;
  return (k_stride << log2_filter_element_size) + bias_element_size +
         extra_bytes;
}

void xnn_pack_qs4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const size_t extra_bytes = extra_data0_element_size + extra_data1_element_size;
  const size_t weights_stride = xnn_packed_stride_qs8_weights_and_biases(
      gemm_config, input_channels, k_stride, extra_bytes);
  return pack_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      weights_stride,
      (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w,
      (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w,
      accumulator_init, weights, init_extra_data0_fn, extra_data0,
      extra_data0_element_size, init_extra_data1_fn, extra_data1,
      extra_data1_element_size, packed_weights_ptr, extra_bytes, params);
}

size_t xnn_packed_stride_qu8_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t unused_k, size_t k_stride,
    size_t extra_bytes) {
  const size_t bias_element_size = sizeof(int32_t);
  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_INT8_T;
  return (k_stride << log2_filter_element_size) + bias_element_size +
         extra_bytes;
}

void xnn_pack_qu8_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const size_t extra_bytes =
      extra_data0_element_size + extra_data1_element_size;
  const size_t weights_stride = xnn_packed_stride_qs8_weights_and_biases(
      gemm_config, input_channels, k_stride, extra_bytes);
  return pack_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      weights_stride, (xnn_packw_gemm_gio_ukernel_fn)xnn_pack_qu8_gemm_gio_w,
      (xnn_packw_gemm_goi_ukernel_fn)xnn_pack_qu8_gemm_goi_w, accumulator_init,
      weights, init_extra_data0_fn, extra_data0, extra_data0_element_size,
      init_extra_data1_fn, extra_data1, extra_data1_element_size,
      packed_weights_ptr, extra_bytes, params);
}

#if XNN_ENABLE_KLEIDIAI
size_t xnn_packed_stride_kai_qs4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t k, size_t unused_k_stride,
    size_t extra_bytes) {
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  return kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4cxp_qsu4cxs1s0(k, /*nr=*/1,
                                                                   kr, sr);
}

void xnn_pack_kai_qs4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const struct xnn_qs8_qc4w_packing_params* xnn_params =
      reinterpret_cast<const struct xnn_qs8_qc4w_packing_params*>(params);

  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    // Repack the packing params.
    struct kai_rhs_pack_kxn_qsi4cxp_qsu4cxs1s0_params kai_params;
    kai_params.lhs_zero_point = xnn_params->input_zero_point;
    kai_params.rhs_zero_point = xnn_params->kernel_zero_point;

    kai_run_rhs_pack_kxn_qsi4cxp_qsu4cxs1s0(
        groups, output_channels, input_channels, nr, kr, sr,
        /*rhs=*/reinterpret_cast<const uint8_t*>(weights),
        /*bias=*/reinterpret_cast<const float*>(extra_data0),
        /*scale=*/reinterpret_cast<const float*>(extra_data1),
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/0,
        &kai_params);
  } else {
    // Repack the packing params.
    struct kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0_params kai_params;
    kai_params.lhs_zero_point = xnn_params->input_zero_point;
    kai_params.rhs_zero_point = xnn_params->kernel_zero_point;

    kai_run_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(
        groups, output_channels, input_channels, nr, kr, sr,
        /*rhs=*/reinterpret_cast<const uint8_t*>(weights),
        /*bias=*/reinterpret_cast<const float*>(extra_data0),
        /*scale=*/reinterpret_cast<const float*>(extra_data1),
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/0,
        &kai_params);
  }
}
#endif  // XNN_ENABLE_KLEIDIAI

void xnn_pack_f32_qs8w_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* k,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const int32_t* b = (const int32_t*) bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                packed_weights[kr_block_offset] = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
              }
            }
            packed_weights += kr;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (float*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f16_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                packed_weights[kr_block_offset] = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
              }
            }
            packed_weights += kr;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = xnn_float16_from_float(b[nr_block_start + nr_block_offset]);
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                packed_weights[kr_block_offset] = xnn_float16_from_float(k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx]);
              }
            }
            packed_weights += kr;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (xnn_float16*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t bzp = (int32_t) ks * (int32_t) kc * izp * (int32_t) params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, bzp);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            int32_t ksum = 0;
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                const uint8_t kv = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
                ksum += (int32_t) kv;
                ((uint8_t*) packed_weights)[kr_block_offset] = kv;
              }
            }
            unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - ksum * izp);
            packed_weights = (uint8_t*) packed_weights + kr;
          }
          packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (int32_t) params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            uint32_t ksum = 0;
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                const int8_t kv = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
                ksum += (uint32_t) kv;
                ((int8_t*) packed_weights)[kr_block_offset] = kv;
              }
            }
            unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
            packed_weights = (int8_t*) packed_weights + kr;
          }
          packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (int32_t) params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            uint32_t ksum = 0;
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                const int8_t kv = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
                ksum += (uint32_t) kv;
                ((int8_t*) packed_weights)[kr_block_offset] = kv;
              }
            }
            unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
            packed_weights = (int8_t*) packed_weights + kr;
          }
          packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] = k[ki * g * nc + (nr_block_start + nr_block_offset)];
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (float*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_f16_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] = k[ki * g * nc + (nr_block_start + nr_block_offset)];
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_f32_to_f16_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = xnn_float16_from_float(b[nr_block_start + nr_block_offset]);
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] = xnn_float16_from_float(k[ki * g * nc + (nr_block_start + nr_block_offset)]);
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (xnn_float16*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_qu8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t bzp = (int32_t) ks * izp * (int32_t) params->kernel_zero_point;
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, bzp);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            const uint8_t kv = k[ki * g * nc + (nr_block_start + nr_block_offset)];
            ((uint8_t*) packed_weights)[nr_block_offset * kr] = kv;
            unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - (int32_t) kv * izp);
          }
          packed_weights = (uint8_t*) packed_weights + nr * kr;
        }
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void pack_qs8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  int32_t zero_point_offset,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const uint32_t izp = (uint32_t) params->input_zero_point + zero_point_offset;
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != nullptr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            const int8_t kv = k[ki * g * nc + (nr_block_start + nr_block_offset)];
            ((int8_t*) packed_weights)[nr_block_offset * kr] = kv;
            unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - (uint32_t) kv * izp);
          }
          packed_weights = (int8_t*) packed_weights + nr * kr;
        }
      }
      packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_qs8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  pack_qs8_conv_kgo_w(g, nc, ks, nr, kr, sr, k, b, scale, packed_weights,
                      extra_bytes, /*zero_point_offset=*/0, params);
}

void xnn_pack_qs8_to_qu8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  pack_qs8_conv_kgo_w(g, nc, ks, nr, kr, sr, k, b, scale, packed_weights,
                      extra_bytes, /*zero_point_offset=*/128, params);
}

void xnn_pack_f32_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          if XNN_LIKELY(b != nullptr) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
            }
          }
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      packed_weights[kr_block_offset] = k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx];
                    }
                  }
                  packed_weights += kr;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = reinterpret_cast<float*>((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_f16_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          if XNN_LIKELY(b != nullptr) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
            }
          }
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      packed_weights[kr_block_offset] = k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx];
                    }
                  }
                  packed_weights += kr;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = reinterpret_cast<uint16_t*>((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_f32_to_f16_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          if XNN_LIKELY(b != nullptr) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              packed_weights[nr_block_offset] = xnn_float16_from_float(b[nr_block_start + nr_block_offset]);
            }
          }
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      packed_weights[kr_block_offset] = xnn_float16_from_float(k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx]);
                    }
                  }
                  packed_weights += kr;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = reinterpret_cast<xnn_float16*>((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void pack_qs8_deconv_goki_w(
  size_t groups,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  int32_t zero_point_offset,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params)
{
  assert(groups != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point + zero_point_offset;
  for (size_t i = 0; i < groups; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          int32_t* packed_b = (int32_t*) packed_weights;
          if XNN_LIKELY(b != 0) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
              packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
            }
          } else {
            size_t n = nr_block_size;
            do {
              unaligned_store_s32(packed_weights, 0);
              packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
            } while (--n != 0);
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  uint32_t ksum = 0;
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      const int8_t kv = k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx];
                      ksum += (uint32_t) kv;
                      ((int8_t*) packed_weights)[kr_block_offset] = kv;
                    }
                  }
                  unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
                  packed_weights = (int8_t*) packed_weights + kr;
                }
                packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_qs8_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params)
{
  pack_qs8_deconv_goki_w(g, nc, kh, kw, kc, sh, sw, nr, kr, sr, k, b, scale,
                         packed_weights, extra_bytes, /*zero_point_offset=*/0, subconv_params, params);
}

void xnn_pack_qs8_to_qu8_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params)
{
  pack_qs8_deconv_goki_w(g, nc, kh, kw, kc, sh, sw, nr, kr, sr, k, b, scale,
                         packed_weights, extra_bytes, /*zero_point_offset=*/128, subconv_params, params);
}

void xnn_pack_qu8_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t kzp = (int32_t) params->kernel_zero_point;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        const int32_t bzp = (int32_t) divide_round_up(kh - oy, sh) * (int32_t) divide_round_up(kw - ox, sw) * (int32_t) kc * izp * kzp;
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          int32_t* packed_b = (int32_t*) packed_weights;
          if XNN_LIKELY(b != 0) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
              packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
            }
          } else {
            size_t n = nr_block_size;
            do {
              unaligned_store_s32(packed_weights, bzp);
              packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
            } while (--n != 0);
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  int32_t ksum = 0;
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      const uint8_t kv = k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx];
                      ksum += (int32_t) kv;
                      ((uint8_t*) packed_weights)[kr_block_offset] = kv;
                    }
                  }
                  unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - ksum * izp);
                  packed_weights = (uint8_t*) packed_weights + kr;
                }
                packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = reinterpret_cast<void*>((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nc;
    }
  }
}

// Helper function to advance x and y indices.
inline static void advance_x_y(size_t h, size_t* x, size_t* y) {
  if (++*y == h) {
    *y = 0;
    ++*x;
  }
}

void xnn_pack_f32_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (float*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (float*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f16_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f32_to_f16_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* k,
  const float* b,
  const void* scale,
  xnn_float16* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = xnn_float16_from_float(b[cr_block_start + cr_block_offset]);
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = xnn_float16_zero();
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = xnn_float16_from_float(b[cr_block_start + cr_block_offset]);
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = xnn_float16_zero();
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (xnn_float16*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (xnn_float16*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}


void xnn_pack_qu8_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t boff = (int32_t) h * (int32_t) w * izp * (int32_t) params->kernel_zero_point;
  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, boff + b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, boff);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }

      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
            unaligned_indexed_store_s32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_s32(packed_b, cr_block_offset) - (int32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, boff + b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, boff);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
            unaligned_indexed_store_s32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_s32(packed_b, cr_block_offset) - (int32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_tile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_subtile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_qs8_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  const uint32_t izp = (uint32_t) params->input_zero_point;
  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }

      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
            unaligned_indexed_store_u32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_u32(packed_b, cr_block_offset) - (uint32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
      // If unipass and QC8, we need to pack extra bytes for scale values here.
      if (middle_pass_tile == 0) {
        packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
      }
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
            unaligned_indexed_store_u32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_u32(packed_b, cr_block_offset) - (uint32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
      // If unipass and QC8, we need to pack extra bytes for scale values here.
      if (middle_pass_tile == 0) {
        packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
      }
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_tile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_subtile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f32_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (float*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (float*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f16_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f32_to_f16_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* k,
  const float* b,
  const void* scale,
  xnn_float16* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = xnn_float16_from_float(b[cr_block_start + cr_block_offset]);
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = xnn_float16_zero();
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = xnn_float16_from_float(b[cr_block_start + cr_block_offset]);
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = xnn_float16_zero();
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (xnn_float16*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const xnn_float16 kv = xnn_float16_from_float(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (xnn_float16*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_qu8_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t boff = (int32_t) h * (int32_t) w * izp * (int32_t) params->kernel_zero_point;
  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, boff + b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, boff);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }

      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
            unaligned_indexed_store_s32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_s32(packed_b, cr_block_offset) - (int32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, boff + b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, boff);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
            unaligned_indexed_store_s32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_s32(packed_b, cr_block_offset) - (int32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_tile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_subtile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_qs8_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  const uint32_t izp = (uint32_t) params->input_zero_point;
  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }

      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
            unaligned_indexed_store_u32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_u32(packed_b, cr_block_offset) - (uint32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
      // If unipass and QC8, we need to pack extra bytes for scale values here.
      if (middle_pass_tile == 0) {
        packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
      }
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != nullptr) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
            unaligned_indexed_store_u32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_u32(packed_b, cr_block_offset) - (uint32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
      // If unipass and QC8, we need to pack extra bytes for scale values here.
      if (middle_pass_tile == 0) {
        packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
      }
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_tile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_subtile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f32_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  float* packed_weights,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
    }
    k += nc * kc;
  } while (--g != 0);
}

void xnn_pack_f16_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  uint16_t* packed_weights,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
    }
    k += nc * kc;
  } while (--g != 0);
}

void xnn_pack_f32_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const float* k,
  const float* b,
  float* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != nullptr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
        *packed_weights++ = b[min(nr_block_offset, nr_block_size - 1)];
      }
    } else {
      size_t n = nr;
      do {
        *packed_weights++ = 0.0f;
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
            *packed_weights++ = k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) * kh + ky) * kw + kx) * kc + c];
          }
        }
      }
    }
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nr;
    }
  }
}

void xnn_pack_f32_to_f16_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const float* k,
  const float* b,
  xnn_float16* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != nullptr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
        *packed_weights++ = xnn_float16_from_float(b[min(nr_block_offset, nr_block_size - 1)]);
      }
    } else {
      size_t n = nr;
      do {
        *packed_weights++ = xnn_float16_zero();
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
            *packed_weights++ = xnn_float16_from_float(k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) * kh + ky) * kw + kx) * kc + c]);
          }
        }
      }
    }
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nr;
    }
  }
}

void xnn_pack_f16_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != nullptr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
        *packed_weights++ = b[min(nr_block_offset, nr_block_size - 1)];
      }
    } else {
      size_t n = nr;
      do {
        *packed_weights++ = 0;
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
            *packed_weights++ = k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) * kh + ky) * kw + kx) * kc + c];
          }
        }
      }
    }
    if XNN_UNPREDICTABLE(b != nullptr) {
      b += nr;
    }
  }
}

void xnn_pack_f32_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const float* k,
  const float* b,
  float* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != nullptr) {
      *packed_weights = *b++;
    } else {
      *packed_weights = 0.0f;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = k[g * kernel_size + i];
    }
  }
}

void xnn_pack_f32_to_f16_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const float* k,
  const float* b,
  xnn_float16* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != nullptr) {
      *packed_weights = xnn_float16_from_float(*b++);
    } else {
      *packed_weights = xnn_float16_zero();
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = xnn_float16_from_float(k[g * kernel_size + i]);
    }
  }
}

void xnn_pack_f16_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != nullptr) {
      *packed_weights = *b++;
    } else {
      *packed_weights = 0;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = k[g * kernel_size + i];
    }
  }
}

void xnn_pack_f32_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const float* k,
  const float* b,
  float* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != nullptr) {
      *packed_weights = *b++;
    } else {
      *packed_weights = 0.0f;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = k[i * groups + g];
    }
  }
}

void xnn_pack_f16_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != nullptr) {
      *packed_weights = *b++;
    } else {
      *packed_weights = 0;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = k[i * groups + g];
    }
  }
}

void xnn_pack_f32_to_f16_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const float* k,
  const float* b,
  xnn_float16* packed_weights,
  const void* params)
{
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != nullptr) {
      *packed_weights = xnn_float16_from_float(*b++);
    } else {
      *packed_weights = xnn_float16_zero();
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = xnn_float16_from_float(k[i * groups + g]);
    }
  }
}


void xnn_pack_f32_vmulcaddc_w(
  size_t c,
  size_t cr,
  const float* s,
  const float* b,
  float* packed_weights,
  const void* params)
{
  assert(s != nullptr);
  assert(packed_weights != nullptr);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
      *packed_weights++ = s[cr_block_start + cr_block_offset];
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY(b != nullptr) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_weights++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_weights++ = 0.0f;
      } while (--n != 0);
    }
    packed_weights += cr - cr_block_size;
  }
}

void xnn_pack_f16_vmulcaddc_w(
  size_t c,
  size_t cr,
  const uint16_t* s,
  const uint16_t* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(s != nullptr);
  assert(packed_weights != nullptr);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
      *packed_weights++ = s[cr_block_start + cr_block_offset];
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY(b != nullptr) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_weights++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_weights++ = 0;
      } while (--n != 0);
    }
    packed_weights += cr - cr_block_size;
  }
}

void xnn_pack_f32_to_f16_vmulcaddc_w(
  size_t c,
  size_t cr,
  const float* s,
  const float* b,
  xnn_float16* packed_weights,
  const void* params)
{
  assert(s != nullptr);
  assert(packed_weights != nullptr);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
      *packed_weights++ = xnn_float16_from_float(s[cr_block_start + cr_block_offset]);
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY(b != nullptr) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_weights++ = xnn_float16_from_float(b[cr_block_start + cr_block_offset]);
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_weights++ = xnn_float16_zero();
      } while (--n != 0);
    }
    packed_weights += cr - cr_block_size;
  }
}

void xnn_pack_f32_prelu_w(
  size_t input_channels,
  size_t slope_channels,
  const float* s,
  float* packed_weights)
{
  assert(s != nullptr);
  assert(packed_weights != nullptr);
  assert(slope_channels == input_channels || slope_channels == 1);

  if (slope_channels == 1) {
    do {
      *packed_weights++ = *s;
    } while (--input_channels != 0);
  } else {
    memcpy(packed_weights, s, slope_channels * sizeof(float));
  }
}

void xnn_pack_f16_prelu_w(
  size_t input_channels,
  size_t slope_channels,
  const uint16_t* s,
  uint16_t* packed_weights)
{
  assert(s != nullptr);
  assert(packed_weights != nullptr);
  assert(slope_channels == input_channels || slope_channels == 1);

  if (slope_channels == 1) {
    do {
      *packed_weights++ = *s;
    } while (--input_channels != 0);
  } else {
    memcpy(packed_weights, s, slope_channels * sizeof(uint16_t));
  }
}

void xnn_pack_f32_to_f16_prelu_w(
  size_t input_channels,
  size_t slope_channels,
  const float* s,
  xnn_float16* packed_weights)
{
  assert(s != nullptr);
  assert(packed_weights != nullptr);
  assert(slope_channels == input_channels || slope_channels == 1);

  if (slope_channels == 1) {
    xnn_float16 v =  xnn_float16_from_float(*s);
    for (size_t i = 0; i < input_channels; ++i) {
      packed_weights[i] = v;
    }
  } else {
    do {
      *packed_weights++ = xnn_float16_from_float(*s++);
    } while (--input_channels != 0);
  }
}

void xnn_analyze_f32_spmm_w(
  size_t group_output_channels,
  size_t group_input_channels,
  const float* kernel,
  struct xnn_spmm_packing_params* params)
{
  assert(kernel != nullptr);
  assert(params != nullptr);

  // Count number of non-zero values.
  size_t num_nonzeroes = 0;
  size_t num_nonzero_blocks2 = 0;
  size_t num_nonzero_blocks4 = 0;
  for (size_t oc = 0; oc < round_down_po2(group_output_channels, 4); oc += 4) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero = (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
      const size_t row1_nonzero = (size_t) (kernel[(oc + 1) * group_input_channels + ic] != 0.0f);
      const size_t row2_nonzero = (size_t) (kernel[(oc + 2) * group_input_channels + ic] != 0.0f);
      const size_t row3_nonzero = (size_t) (kernel[(oc + 3) * group_input_channels + ic] != 0.0f);
      num_nonzeroes += row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
      num_nonzero_blocks4 += (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
    }
  }
  const size_t num_block4_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 4); oc < round_down_po2(group_output_channels, 2); oc += 2) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero = (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
      const size_t row1_nonzero = (size_t) (kernel[(oc + 1) * group_input_channels + ic] != 0.0f);
      num_nonzeroes += row0_nonzero + row1_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
    }
  }
  const size_t num_block2_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 2); oc < group_output_channels; oc++) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      num_nonzeroes += (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
    }
  }
  params->num_nonzeroes = num_nonzeroes;
  params->num_nonzero_blocks2 = num_nonzero_blocks2;
  params->num_nonzero_blocks4 = num_nonzero_blocks4;
  params->num_block2_nonzeroes = num_block2_nonzeroes;
  params->num_block4_nonzeroes = num_block4_nonzeroes;
}

void xnn_analyze_f16_spmm_w(
  size_t group_output_channels,
  size_t group_input_channels,
  const xnn_float16* kernel,
  struct xnn_spmm_packing_params* params)
{
  assert(kernel != nullptr);
  assert(params != nullptr);

  // Count number of non-zero values.
  size_t num_nonzeroes = 0;
  size_t num_nonzero_blocks2 = 0;
  size_t num_nonzero_blocks4 = 0;
  for (size_t oc = 0; oc < round_down_po2(group_output_channels, 4); oc += 4) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero = (size_t) !xnn_float16_is_zero(kernel[oc * group_input_channels + ic]);
      const size_t row1_nonzero = (size_t) !xnn_float16_is_zero(kernel[(oc + 1) * group_input_channels + ic]);
      const size_t row2_nonzero = (size_t) !xnn_float16_is_zero(kernel[(oc + 2) * group_input_channels + ic]);
      const size_t row3_nonzero = (size_t) !xnn_float16_is_zero(kernel[(oc + 3) * group_input_channels + ic]);
      num_nonzeroes += row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
      num_nonzero_blocks4 += (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
    }
  }
  const size_t num_block4_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 4); oc < round_down_po2(group_output_channels, 2); oc += 2) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero = (size_t) !xnn_float16_is_zero(kernel[oc * group_input_channels + ic]);
      const size_t row1_nonzero = (size_t) !xnn_float16_is_zero(kernel[(oc + 1) * group_input_channels + ic]);
      num_nonzeroes += row0_nonzero + row1_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
    }
  }
  const size_t num_block2_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 2); oc < group_output_channels; oc++) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      num_nonzeroes += (size_t) !xnn_float16_is_zero(kernel[oc * group_input_channels + ic]);
    }
  }
  params->num_nonzeroes = num_nonzeroes;
  params->num_nonzero_blocks2 = num_nonzero_blocks2;
  params->num_nonzero_blocks4 = num_nonzero_blocks4;
  params->num_block2_nonzeroes = num_block2_nonzeroes;
  params->num_block4_nonzeroes = num_block4_nonzeroes;
}

enum xnn_status xnn_pack_f32_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const float* kernel,
  const float* bias,
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  float* nonzero_values,
  size_t* first_input_channel)
{
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0; ocb < round_down_po2(group_output_channels, output_channels_block_size); ocb += output_channels_block_size) {
    if XNN_LIKELY(bias != nullptr) {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = bias[ocb + oco];
      }
    } else {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = 0.0f;
      }
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |= (kernel[(ocb + oco) * group_input_channels + ic] != 0.0f);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = kernel[(ocb + oco) * group_input_channels + ic];
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc = round_down_po2(group_output_channels, output_channels_block_size); oc < group_output_channels; oc++) {
    if XNN_LIKELY(bias != nullptr) {
      *nonzero_values++ = bias[oc];
    } else {
      *nonzero_values++ = 0.0f;
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const float weight = kernel[oc * group_input_channels + ic];
      if (weight != 0.0f) {
        *nonzero_values++ = weight;
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t) ((uint64_t) first_ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
    if (diff != (int64_t) (int32_t) diff) {
      xnn_log_error("failed to convert kernel to sparse representation: "
        "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t) diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}


enum xnn_status xnn_pack_f32_to_f16_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const float* kernel,
  const float* bias,
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  xnn_float16* nonzero_values,  // fp16 values
  size_t* first_input_channel)
{
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0; ocb < round_down_po2(group_output_channels, output_channels_block_size); ocb += output_channels_block_size) {
    if XNN_LIKELY(bias != nullptr) {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = xnn_float16_from_float(bias[ocb + oco]);
      }
    } else {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = xnn_float16_zero();
      }
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |= (kernel[(ocb + oco) * group_input_channels + ic] != 0.0f);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = xnn_float16_from_float(kernel[(ocb + oco) * group_input_channels + ic]);
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc = round_down_po2(group_output_channels, output_channels_block_size); oc < group_output_channels; oc++) {
    if XNN_LIKELY(bias != nullptr) {
      *nonzero_values++ = xnn_float16_from_float(bias[oc]);
    } else {
      *nonzero_values++ = xnn_float16_zero();
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const float weight = kernel[oc * group_input_channels + ic];
      if (weight != 0.0f) {
        *nonzero_values++ = xnn_float16_from_float(weight);
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t) ((uint64_t) first_ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
    if (diff != (int64_t) (int32_t) diff) {
      xnn_log_error("failed to convert kernel to sparse representation: "
        "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t) diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}

enum xnn_status xnn_pack_f16_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const xnn_float16* kernel,  // fp16 values
  const xnn_float16* bias,  // fp16 values
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  xnn_float16* nonzero_values,  // fp16 values
  size_t* first_input_channel)
{
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0; ocb < round_down_po2(group_output_channels, output_channels_block_size); ocb += output_channels_block_size) {
    if XNN_LIKELY(bias != nullptr) {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = bias[ocb + oco];
      }
    } else {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = xnn_float16_zero();
      }
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |= !xnn_float16_is_zero(kernel[(ocb + oco) * group_input_channels + ic]);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = kernel[(ocb + oco) * group_input_channels + ic];
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc = round_down_po2(group_output_channels, output_channels_block_size); oc < group_output_channels; oc++) {
    if XNN_LIKELY(bias != nullptr) {
      *nonzero_values++ = bias[oc];
    } else {
      *nonzero_values++ = xnn_float16_zero();
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const xnn_float16 weight = kernel[oc * group_input_channels + ic];
      if (!xnn_float16_is_zero(weight)) {
        *nonzero_values++ = weight;
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t) ((uint64_t) first_ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
    if (diff != (int64_t) (int32_t) diff) {
      xnn_log_error("failed to convert kernel to sparse representation: "
        "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t) diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}

}  // extern "C"
