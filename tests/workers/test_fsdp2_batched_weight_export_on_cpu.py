# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from verl.workers.engine.fsdp import transformer_impl


class _ExportTracker:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.events = []


class _FakeFSDPUnit(nn.Module):
    def __init__(self, name: str, tracker: _ExportTracker):
        super().__init__()
        self.name = name
        self.tracker = tracker
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
        self.tied_weight = self.weight
        self.register_buffer("scale", torch.tensor(2.0))
        self.register_buffer("scratch", torch.tensor(3.0), persistent=False)

    def unshard(self):
        self.tracker.events.append((self.name, "unshard"))
        self.tracker.active += 1
        self.tracker.max_active = max(self.tracker.max_active, self.tracker.active)

    def reshard(self):
        self.tracker.events.append((self.name, "reshard"))
        self.tracker.active -= 1


def _build_nested_units():
    tracker = _ExportTracker()
    root = _FakeFSDPUnit("root", tracker)
    root.child = _FakeFSDPUnit("child", tracker)
    root.unwrapped = nn.Linear(2, 1, bias=False)
    return root, tracker


def _mock_fsdp2(monkeypatch):
    monkeypatch.setattr(
        transformer_impl,
        "fsdp_version",
        lambda module: 2 if isinstance(module, _FakeFSDPUnit) else 0,
    )


def test_batched_export_materializes_one_fsdp_unit_at_a_time(monkeypatch):
    root, tracker = _build_nested_units()
    _mock_fsdp2(monkeypatch)

    exported = dict(transformer_impl._iter_fsdp2_unsharded_state(root, root, torch.device("cpu")))

    assert set(exported) == {
        "weight",
        "tied_weight",
        "scale",
        "unwrapped.weight",
        "child.weight",
        "child.tied_weight",
        "child.scale",
    }
    assert "scratch" not in exported
    assert exported["weight"].dtype == torch.bfloat16
    assert exported["unwrapped.weight"].dtype == torch.bfloat16
    assert exported["scale"].dtype == torch.float32
    assert torch.equal(exported["weight"], exported["tied_weight"])
    assert tracker.max_active == 1
    assert tracker.active == 0
    assert tracker.events == [
        ("root", "unshard"),
        ("root", "reshard"),
        ("child", "unshard"),
        ("child", "reshard"),
    ]


def test_batched_export_reshards_when_generator_is_closed(monkeypatch):
    root, tracker = _build_nested_units()
    _mock_fsdp2(monkeypatch)

    exported = transformer_impl._iter_fsdp2_unsharded_state(root, root, torch.device("cpu"))
    next(exported)
    assert tracker.active == 1

    exported.close()

    assert tracker.active == 0
    assert tracker.events == [("root", "unshard"), ("root", "reshard")]


def test_engine_uses_batched_export_without_whole_model_staging(monkeypatch):
    root, tracker = _build_nested_units()
    _mock_fsdp2(monkeypatch)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        transformer_impl,
        "load_fsdp_model_to_gpu",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("whole-model staging must be skipped")),
    )
    monkeypatch.setattr(
        transformer_impl,
        "offload_fsdp_model_to_cpu",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("whole-model offload must be skipped")),
    )

    engine = object.__new__(transformer_impl.FSDPEngine)
    engine.module = root
    engine._uses_fsdp2_cpu_offload_policy = False
    engine._is_offload_param = True
    engine._qat_enabled = False
    engine.model_config = SimpleNamespace(lora={"merge": False})

    exported, peft_config = engine.get_per_tensor_param()

    assert peft_config is None
    assert "child.weight" in dict(exported)
    assert tracker.active == 0
