#!/usr/bin/env bash
# Docker entrypoint script.

set -e

mkdir -p /opt/nsight/bin && ln -sf "$NSYS_ROOT/target-linux-x64/nsys" /opt/nsight/bin/nsys
export PATH=/opt/nsight/bin:$PATH

exec "$@"

