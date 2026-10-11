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

"""NVTX instrumentation gate and capture-window tests.

These are pure logic: the CUDA entry points are monkeypatched, so the suite is
CPU-safe. The regression that matters is that *uncaptured* calls synchronize
zero times -- a capture decorator that syncs unconditionally is a permanent
latency tax on the serving path.
"""

import importlib

import pytest


def _reload_nvtx(monkeypatch, **env):
    """Re-import the module so its import-time env gate is re-evaluated."""
    for key in ("GR00T_NVTX", "NSYS_WARMUP"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import gr00t.utils.nvtx as nvtx

    return importlib.reload(nvtx)


@pytest.fixture
def counters(monkeypatch):
    """Monkeypatch the CUDA profiler surface and count every call."""
    counts = {"sync": 0, "start": 0, "stop": 0, "push": 0, "pop": 0}

    import torch

    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a, **k: counts.__setitem__("sync", counts["sync"] + 1)
    )
    monkeypatch.setattr(
        torch.cuda.profiler,
        "start",
        lambda *a, **k: counts.__setitem__("start", counts["start"] + 1),
    )
    monkeypatch.setattr(
        torch.cuda.profiler, "stop", lambda *a, **k: counts.__setitem__("stop", counts["stop"] + 1)
    )
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_push",
        lambda *a, **k: counts.__setitem__("push", counts["push"] + 1),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_pop", lambda *a, **k: counts.__setitem__("pop", counts["pop"] + 1)
    )
    return counts


def test_gate_defaults_off(monkeypatch, counters):
    """With no env var set, every helper is inert and touches no CUDA call."""
    nvtx = _reload_nvtx(monkeypatch)

    assert nvtx.nvtx_enabled() is False
    with nvtx.nvtx_phase("prep.processor"):
        pass
    nvtx.nvtx_push("step")
    nvtx.nvtx_pop()

    assert counters == {"sync": 0, "start": 0, "stop": 0, "push": 0, "pop": 0}


@pytest.mark.parametrize("value", ["0", "", "false", "no", "FALSE", "No"])
def test_falsey_values_keep_gate_off(monkeypatch, value):
    nvtx = _reload_nvtx(monkeypatch, GR00T_NVTX=value)
    assert nvtx.nvtx_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_truthy_values_enable_gate(monkeypatch, value):
    nvtx = _reload_nvtx(monkeypatch, GR00T_NVTX=value)
    assert nvtx.nvtx_enabled() is True


def test_push_pop_balanced_when_enabled(monkeypatch, counters):
    nvtx = _reload_nvtx(monkeypatch, GR00T_NVTX="1")

    nvtx.nvtx_push("step")
    nvtx.nvtx_pop()

    assert counters["push"] == 1
    assert counters["pop"] == 1
    # The range helpers must never synchronize: doing so serialises CPU and GPU
    # and manufactures exactly the idle gaps profiling is meant to find.
    assert counters["sync"] == 0


def test_capture_window_is_a_noop_when_disabled(monkeypatch, counters):
    """Gate off: the decorator returns the function untouched."""
    nvtx = _reload_nvtx(monkeypatch)

    def fn():
        return "ok"

    wrapped = nvtx.nsys_profile(fn)
    assert wrapped is fn
    for _ in range(20):
        wrapped()
    assert counters == {"sync": 0, "start": 0, "stop": 0, "push": 0, "pop": 0}


def test_capture_window_fires_only_on_warm_calls(monkeypatch, counters):
    """Only calls [warmup, warmup + _CAPTURE_REQUESTS) open a capture range."""
    nvtx = _reload_nvtx(monkeypatch, GR00T_NVTX="1", NSYS_WARMUP="3")

    captured_at = []

    @nvtx.nsys_profile
    def fn(index):
        captured_at.append((index, counters["start"]))
        return index

    for i in range(10):
        assert fn(i) == i

    # Exactly _CAPTURE_REQUESTS captures, starting after the warmup calls.
    assert counters["start"] == nvtx._CAPTURE_REQUESTS == 2
    assert counters["stop"] == 2
    # Two syncs per captured call (before start, after stop), none otherwise.
    assert counters["sync"] == 2 * 2

    # Calls 3 and 4 are the captured ones; the profiler was already started
    # when the body ran, which is what makes the range cover the work.
    started_during = [index for index, started in captured_at if started > 0]
    assert started_during[:2] == [3, 4]


def test_uncaptured_calls_never_synchronize(monkeypatch, counters):
    """The regression that matters: no sync tax outside the capture window."""
    nvtx = _reload_nvtx(monkeypatch, GR00T_NVTX="1", NSYS_WARMUP="2")

    @nvtx.nsys_profile
    def fn():
        return None

    # Drain past the capture window, then reset and keep calling.
    for _ in range(2 + nvtx._CAPTURE_REQUESTS):
        fn()
    counters["sync"] = 0

    for _ in range(50):
        fn()

    assert counters["sync"] == 0
    assert counters["start"] == nvtx._CAPTURE_REQUESTS


def test_decorator_preserves_identity(monkeypatch):
    nvtx = _reload_nvtx(monkeypatch, GR00T_NVTX="1")

    @nvtx.nsys_profile
    def documented(a, b=2):
        """Docstring survives."""
        return a + b

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring survives."
    assert documented(1, b=5) == 6
