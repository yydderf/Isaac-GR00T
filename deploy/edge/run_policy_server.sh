#!/bin/bash
# python -m cosmos_framework.scripts.action_policy_server_robolab \
#   --checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID \
#   --port 8000 \
#   --format-prompt-as-json True \
#   --guidance-interval 960 1001 \
#   --num-steps 10
#

GR00T_NVTX=1 NSYS_WARMUP=3 \
MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '-') \
POLICY=gr00t_n1d7 DATETIME=$(date +"%Y%m%d_%H%M") \
    nsys profile \
    --output /workspace/Isaac-GR00T/outputs/nsys/%q{MODEL}_%q{POLICY}_%q{DATETIME}_%p --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set=ga10y \
    --gpu-metrics-frequency=2000 \
    --cuda-graph-trace=node \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi --capture-range-end=repeat:2 \
    --sample=process-tree --cpuctxsw=process-tree \
    python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.7-DROID \
        --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
        --use-sim-policy-wrapper

