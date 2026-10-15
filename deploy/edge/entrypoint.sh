#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


# Docker entrypoint script.

set -e

NSYS_ROOT="${NSYS_ROOT:-/opt/nvidia/nsight-systems/2026.3.1}"

if [ -d "$NSYS_ROOT" ]; then
  if [ -x "$NSYS_ROOT/bin/nsys" ]; then
    export PATH="$NSYS_ROOT/bin:$PATH"
  else
    NSYS_TARGET="$(ls -d "$NSYS_ROOT"/target-linux-* 2>/dev/null | head -1)"
    [ -n "$NSYS_TARGET" ] && export PATH="$NSYS_TARGET:$PATH"
  fi
fi

exec "$@"
