#!/bin/sh
################################ RISC-V Vector ################################
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m1 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x1v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m1 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x1v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m1 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x1v-minmax-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m1 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x1v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m1 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x1v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m1 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x1v-minmax-rvv.c &


tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=8 -D NR=m2 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-8x2v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=8 -D NR=m2 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-8x2v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=8 -D NR=m2 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-8x2v-minmax-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m2 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x2v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m2 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x2v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m2 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x2v-minmax-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m2 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x2v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m2 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x2v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m2 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x2v-minmax-rvv.c &


tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m4 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x4v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m4 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x4v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=7 -D NR=m4 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-7x4v-minmax-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m4 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x4v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m4 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x4v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m4 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x4v-minmax-rvv.c &


tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=4 -D NR=m8 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-4x8v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=4 -D NR=m8 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-4x8v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=4 -D NR=m8 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-4x8v-minmax-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=3 -D NR=m8 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-3x8v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=3 -D NR=m8 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-3x8v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=3 -D NR=m8 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-3x8v-minmax-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m8 -D ACTIVATION=LINEAR -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x8v-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m8 -D ACTIVATION=RELU   -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x8v-relu-rvv.c &
tools/xngen src/f32-gemm-transposea/MRxNRv-pruned-rvv.c.in -D MR=1 -D NR=m8 -D ACTIVATION=MINMAX -D DATATYPE=F32 -o src/f32-gemm-transposea/gen/f32-gemm-transposea-pruned-1x8v-minmax-rvv.c &

wait