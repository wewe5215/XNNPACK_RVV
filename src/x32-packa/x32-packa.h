// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, unroll)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, nr, kr, sr, kblock, nr_scale

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packa_gemm_ukernel_x1v__rvv_u8, 1, 1, 1, 8, (xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packa_gemm_ukernel_x2v__rvv_u8, 1, 1, 1, 8, (xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packa_gemm_ukernel_x4v__rvv_u8, 1, 1, 1, 8, (xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packa_gemm_ukernel_x8v__rvv_u8, 2, 1, 1, 8, (xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV()


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

