
# XNNPACK_RVV
XNNPACK_RVV is a based on [XNNPACK](https://github.com/google/XNNPACK) library
## My contributions
- Code implementation for [Efficient Column-Wise N\:M Pruning on RISC-V CPU](https://arxiv.org/abs/2507.17301)
- Integrated the newly developed operators into the AI compiler: [AI_template_RVV_backend](https://github.com/wewe5215/AI_template_RVV_backend/tree/main)
- Implements low-level convolution operators in **CNHW** layout
  - `input_t_average_pooling2d_nhwc_f32`
  - `input_t_max_pooling2d_nhwc_f32`
  - `input_T_convolution2d_nhwc_f32`
  - `input_T_pruned_convolution2d_nhwc_f32_x1v`
  - `input_T_pruned_convolution2d_nhwc_f32_x2v`
  - `input_T_pruned_convolution2d_nhwc_f32_x4v`
  - `input_T_pruned_convolution2d_nhwc_f32_x8v`
- Implements fusion of `im2col` and data packing, located at `XNNPACK_RVV/src/packing.cc`.
  - `im2col_local_avgpool_s2_d1_p0_with_pack_{1, 2, 4, 8}x`
  - `xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x{1, 2, 4}v`
  - `xnn_x32_packa_in_T_gemm_im2col_s1_d1_{2x2v, 4x4v, 4x8v, 8x8v, ....}`
- Provides microkernels with column-wise pruning, located at `src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in`.
## Setup
```
chmod +x scripts/build-local.sh
./scripts/build-local.sh
```
**Compiler requirement:** compile the generated C++ with **Clang ≥ 17.0.2**; older versions lack several RVV v0.12 intrinsics required by XNNPACK.

## Future Work
- Support Convolution 2d with CNHW layout implemented with vmulcaddc
- Support F16 implementation
- currently, the code for Fusion of Im2col and packing is a little bit dirty, as all of the corner cases and conditions are turned into bitwise operation. I will fix this later