# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in NVTX phase ranges for Nsight Systems profiling.

Every helper here is a no-op unless ``GR00T_NVTX=1`` is set in the environment.
The same modules sit on the training path, so the gate defaults to OFF and is
resolved once at import rather than per range in a hot loop.

Placement constraint: NVTX cannot be called from inside a ``torch.compile``
region -- Dynamo cannot place a context manager or ``None`` in the FX graph.
Ranges therefore wrap eager *call sites*, never a compiled callee's body. The
only compile boundary in this repo is ``action_head.model.forward``, which
``scripts/deployment/benchmark_inference.py`` optionally wraps in
``torch.compile(mode="max-autotune")``; the range around it lives in the
denoising loop of ``get_action_with_features``, outside that boundary.

Usage::

    GR00T_NVTX=1 NSYS_WARMUP=10 nsys profile \\
        --trace=cuda,nvtx,cublas,cudnn,osrt \\
        --capture-range=cudaProfilerApi --capture-range-end=repeat:2 \\
        python gr00t/eval/run_gr00t_server.py --model-path ...
"""

import contextlib
import functools
import os
from typing import Callable

import torch


# Resolved once at import: this is a profiling-session switch, not something to
# re-read per range in a hot loop.
_ENABLED: bool = os.environ.get("GR00T_NVTX", "0").strip().lower() not in (
    "0",
    "",
    "false",
    "no",
)

# nullcontext holds no state, so one shared instance is safe across nested and
# repeated `with` blocks. Keeps the disabled path allocation-free.
_NULL: contextlib.nullcontext = contextlib.nullcontext()

# Must match --capture-range-end=repeat:N in the nsys invocation, so nsys shuts
# the session down exactly when the decorator stops opening ranges.
_CAPTURE_REQUESTS = 2


def nvtx_enabled() -> bool:
    """Whether NVTX instrumentation is active for this process."""
    return _ENABLED


def nvtx_phase(name: str):
    """Context manager marking an NVTX range. No-op when disabled.

    Never calls ``torch.cuda.synchronize()``: synchronizing on range boundaries
    serializes CPU and GPU and manufactures exactly the idle gaps you are hunting.
    """
    if not _ENABLED:
        return _NULL
    return torch.cuda.nvtx.range(name)


def nvtx_push(name: str) -> None:
    """Open a range that :func:`nvtx_pop` closes.

    Prefer :func:`nvtx_phase`. This pair is for spans too long to wrap in a
    ``with`` without reindenting the whole body, and is ONLY safe on
    straight-line code: an early return/continue/break, or a routinely-raised
    exception, leaks an unclosed range -- which does not error, it silently
    swallows the rest of the timeline.
    """
    if _ENABLED:
        torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    """Close the innermost range opened by :func:`nvtx_push`."""
    if _ENABLED:
        torch.cuda.nvtx.range_pop()


def nsys_profile(func: Callable) -> Callable:
    """Open a ``cudaProfilerApi`` capture range around the first WARM calls only.

    Starting the profiler on every call would capture requests 1 and 2 -- which
    are warmup, autotuning and allocator growth, not inference -- and would tax
    every later request with an unconditional ``synchronize()``. Uncaptured
    calls here sync zero times and touch the profiler not at all, so the
    decorator is safe to leave in place on a serving path.

    ``NSYS_WARMUP`` (default 10) sets how many calls to skip first.

    When ``GR00T_NVTX`` is off the undecorated function is returned as-is, so a
    CPU-only or non-profiling run never reaches ``torch.cuda.*`` at all.
    """
    if not _ENABLED:
        return func

    warmup = int(os.environ.get("NSYS_WARMUP", "10"))
    state = {"count": 0}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        index = state["count"]
        state["count"] = index + 1
        if not warmup <= index < warmup + _CAPTURE_REQUESTS:
            return func(*args, **kwargs)  # no sync, no profiler calls
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            return func(*args, **kwargs)
        finally:
            torch.cuda.profiler.stop()
            torch.cuda.synchronize()

    return wrapper
