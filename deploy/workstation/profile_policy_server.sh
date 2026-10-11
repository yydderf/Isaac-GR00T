#!/bin/bash
# Profile the GR00T policy server's phases with NVTX + Nsight Systems.
#
# Run INSIDE the container. The container must have been started with:
#   --cap-add=SYS_ADMIN --security-opt seccomp=unconfined
# and the host driver must allow non-admin profiling:
#   cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly   -> must be 0
# otherwise the SM Active / Tensor Active rows come back empty.
#
# The report goes to a bind-mounted path so `docker run --rm` does not eat it.

set -euo pipefail

REPO_ROOT=/workspace/Isaac-GR00T
OUT_DIR="${NSYS_OUT_DIR:-$REPO_ROOT/profiles}"
# RTX 3090 is GA10x. Confirm on any other machine with:
#   nsys profile --gpu-metrics-set=help
GPU_METRICS_SET="${GPU_METRICS_SET:-ga10x}"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

# GR00T_NVTX=1 arms the phase ranges; NSYS_WARMUP skips cold requests so the
# capture lands on steady-state inference, not allocator growth or autotuning.
# --capture-range-end=repeat:2 must match _CAPTURE_REQUESTS in gr00t/utils/nvtx.py.
GR00T_NVTX=1 NSYS_WARMUP="${NSYS_WARMUP:-10}" \
nsys profile \
    --output "$OUT_DIR/gr00t_%q{DATETIME}_%p" --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set="$GPU_METRICS_SET" \
    --gpu-metrics-frequency=10000 \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=repeat:2 \
    --sample=process-tree --cpuctxsw=process-tree \
    python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.7-DROID \
        --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
        --use-sim-policy-wrapper
