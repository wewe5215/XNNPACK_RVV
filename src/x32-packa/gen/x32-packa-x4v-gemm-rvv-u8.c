// Auto-generated file. Do not edit!
//   Template: src/x32-packa/rvv.c.in
//   Generator: tools/xngen
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "xnnpack/packw.h"

void xnn_x32_packa_gemm_ukernel_x4v__rvv_u8(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == __riscv_vsetvlmax_e32m4());
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint32_t* out = packed_weights;
  size_t kc_bstride = kc << 2;

  do {
    const uint32_t* w0 = weights;
    const uint32_t* w1 = w0 + kc;
    const uint32_t* w2 = w1 + kc;
    const uint32_t* w3 = w2 + kc;
    const uint32_t* w4 = w3 + kc;
    const uint32_t* w5 = w4 + kc;
    const uint32_t* w6 = w5 + kc;
    const uint32_t* w7 = w6 + kc;
    long int k = kc;
    uint32_t* out0 = out;
    do {
      size_t vl;
      size_t vlmax = __riscv_vsetvlmax_e32m4();
      if XNN_LIKELY(k >= vlmax) {
        vl = vlmax;
      } else {
        vl = __riscv_vsetvl_e32m4(k);
      }
      size_t n = nc;
      const uint32_t* w_ptr0 = w0;
      const uint32_t* w_ptr1 = w1;
      const uint32_t* w_ptr2 = w2;
      const uint32_t* w_ptr3 = w3;
      const uint32_t* w_ptr4 = w4;
      const uint32_t* w_ptr5 = w5;
      const uint32_t* w_ptr6 = w6;
      const uint32_t* w_ptr7 = w7;
      for (; n >= 8; n -= 8) {
        vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(w_ptr0, vl);
        __riscv_vse32_v_u32m4(out0, v_w0, vl);
        out0 += vlmax;
        vuint32m4_t v_w1 = __riscv_vle32_v_u32m4(w_ptr1, vl);
        __riscv_vse32_v_u32m4(out0, v_w1, vl);
        out0 += vlmax;
        vuint32m4_t v_w2 = __riscv_vle32_v_u32m4(w_ptr2, vl);
        __riscv_vse32_v_u32m4(out0, v_w2, vl);
        out0 += vlmax;
        vuint32m4_t v_w3 = __riscv_vle32_v_u32m4(w_ptr3, vl);
        __riscv_vse32_v_u32m4(out0, v_w3, vl);
        out0 += vlmax;
        vuint32m4_t v_w4 = __riscv_vle32_v_u32m4(w_ptr4, vl);
        __riscv_vse32_v_u32m4(out0, v_w4, vl);
        out0 += vlmax;
        vuint32m4_t v_w5 = __riscv_vle32_v_u32m4(w_ptr5, vl);
        __riscv_vse32_v_u32m4(out0, v_w5, vl);
        out0 += vlmax;
        vuint32m4_t v_w6 = __riscv_vle32_v_u32m4(w_ptr6, vl);
        __riscv_vse32_v_u32m4(out0, v_w6, vl);
        out0 += vlmax;
        vuint32m4_t v_w7 = __riscv_vle32_v_u32m4(w_ptr7, vl);
        __riscv_vse32_v_u32m4(out0, v_w7, vl);
        out0 += vlmax;
        w_ptr0 += (kc << 3);
        w_ptr1 += (kc << 3);
        w_ptr2 += (kc << 3);
        w_ptr3 += (kc << 3);
        w_ptr4 += (kc << 3);
        w_ptr5 += (kc << 3);
        w_ptr6 += (kc << 3);
        w_ptr7 += (kc << 3);
      }

      for (; n >= 4; n -= 4) {
        vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(w_ptr0, vl);
        __riscv_vse32_v_u32m4(out0, v_w0, vl);
        out0 += vlmax;
        vuint32m4_t v_w1 = __riscv_vle32_v_u32m4(w_ptr1, vl);
        __riscv_vse32_v_u32m4(out0, v_w1, vl);
        out0 += vlmax;
        vuint32m4_t v_w2 = __riscv_vle32_v_u32m4(w_ptr2, vl);
        __riscv_vse32_v_u32m4(out0, v_w2, vl);
        out0 += vlmax;
        vuint32m4_t v_w3 = __riscv_vle32_v_u32m4(w_ptr3, vl);
        __riscv_vse32_v_u32m4(out0, v_w3, vl);
        out0 += vlmax;
        w_ptr0 += (kc << 2);
        w_ptr1 += (kc << 2);
        w_ptr2 += (kc << 2);
        w_ptr3 += (kc << 2);
      }

      for (; n >= 1; n -= 1) {
        vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(w_ptr0, vl);
        __riscv_vse32_v_u32m4(out0, v_w0, vl);
        out0 += vlmax;
        w_ptr0 += kc;
      }
      k -= nr;
      w0 += nr;
      w1 += nr;
      w2 += nr;
      w3 += nr;
      w4 += nr;
      w5 += nr;
      w6 += nr;
      w7 += nr;
    } while(k > 0);
  } while (--g != 0);
}
