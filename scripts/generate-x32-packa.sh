#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## RISC-V VECTOR ##################################
tools/xngen src/x32-packa/rvv.c.in -D NR=m1  -o src/x32-packa/gen/x32-packa-x1v-gemm-rvv-u8.c &
tools/xngen src/x32-packa/rvv.c.in -D NR=m2  -o src/x32-packa/gen/x32-packa-x2v-gemm-rvv-u8.c &
tools/xngen src/x32-packa/rvv.c.in -D NR=m4  -o src/x32-packa/gen/x32-packa-x4v-gemm-rvv-u8.c &
tools/xngen src/x32-packa/rvv.c.in -D NR=m8  -o src/x32-packa/gen/x32-packa-x8v-gemm-rvv-u8.c &

wait
