#!/bin/bash
docker run -it --rm \
    --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
    --name gr00t \
    --net host --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_TOKEN=$HF_TOKEN \
    -e NSYS_ROOT=/opt/nvidia/nsight-systems/2026.4.1 \
    -v /opt/nvidia/nsight-systems/2026.4.1:/opt/nvidia/nsight-systems/2026.4.1:ro \
    -v "$(pwd):/workspace/Isaac-GR00T" \
    -v "/srv/models/huggingface:/root/.cache/huggingface" \
    --entrypoint /workspace/Isaac-GR00T/deploy/workstation/entrypoint.sh \
    gr00t \
    bash -c '\
      cd /workspace/Isaac-GR00T && \
      export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" && \
      bash
    '
