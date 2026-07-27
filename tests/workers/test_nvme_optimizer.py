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

import os

import pytest
import torch

from verl.workers.engine.fsdp.nvme_optimizer import NVMeChunkedAdamW, SynchronousNVMeStore


def test_synchronous_store_round_trip_preserves_fp32_and_bf16(tmp_path):
    store = SynchronousNVMeStore(tmp_path / "store")
    store.register("fp32", numel=11, dtype=torch.float32)
    store.register("bf16", numel=7, dtype=torch.bfloat16)

    fp32 = torch.linspace(-2, 2, 11, dtype=torch.float32)
    bf16 = torch.arange(7, dtype=torch.bfloat16)
    store.write("fp32", fp32[:5], offset=0)
    store.write("fp32", fp32[5:], offset=5)
    store.write("bf16", bf16, offset=0)

    assert torch.equal(store.read("fp32", offset=0, count=11), fp32)
    assert torch.equal(store.read("bf16", offset=0, count=7), bf16)


def test_transformer_engine_rejects_nvme_offload_with_fsdp1(tmp_path):
    from verl.workers.config.engine import FSDPEngineConfig, NVMeOffloadConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    engine = object.__new__(FSDPEngine)
    engine.engine_config = FSDPEngineConfig(
        strategy="fsdp",
        nvme_offload=NVMeOffloadConfig(enabled=True, path=str(tmp_path)),
    )
    engine.optimizer_config = FSDPOptimizerConfig()

    with pytest.raises(ValueError, match="strategy=fsdp2"):
        engine._build_optimizer(torch.nn.Linear(2, 2))


def test_synchronous_store_fails_on_short_file(tmp_path):
    store = SynchronousNVMeStore(tmp_path / "store")
    store.register("state", numel=8, dtype=torch.float32)
    os.truncate(tmp_path / "store" / "state.bin", 7)

    with pytest.raises(IOError, match="expected 32"):
        store.read("state", offset=0, count=8)


def test_chunked_adamw_matches_torch_with_missing_gradient(tmp_path):
    torch.manual_seed(7)
    initial_large = torch.randn(300_001, dtype=torch.float32)
    initial_small = torch.randn(17, dtype=torch.float32)
    nvme_parameters = [
        torch.nn.Parameter(initial_large.clone()),
        torch.nn.Parameter(initial_small.clone()),
    ]
    reference_parameters = [
        torch.nn.Parameter(initial_large.clone()),
        torch.nn.Parameter(initial_small.clone()),
    ]
    optimizer = NVMeChunkedAdamW(
        zip(("large", "small"), nvme_parameters, strict=True),
        path=str(tmp_path / "nvme"),
        chunk_size_mb=1,
        lr=3e-4,
        betas=(0.8, 0.95),
        eps=1e-6,
        weight_decay=0.1,
    )
    reference = torch.optim.AdamW(
        reference_parameters,
        lr=3e-4,
        betas=(0.8, 0.95),
        eps=1e-6,
        weight_decay=0.1,
        foreach=False,
    )

    for step in range(3):
        large_gradient = torch.randn_like(initial_large)
        nvme_parameters[0].grad = large_gradient.clone()
        reference_parameters[0].grad = large_gradient.clone()
        if step != 1:
            small_gradient = torch.randn_like(initial_small)
            nvme_parameters[1].grad = small_gradient.clone()
            reference_parameters[1].grad = small_gradient.clone()

        optimizer.step()
        reference.step()
        optimizer.zero_grad()
        reference.zero_grad()

    assert torch.allclose(nvme_parameters[0], reference_parameters[0], rtol=1e-6, atol=1e-7)
    assert torch.allclose(nvme_parameters[1], reference_parameters[1], rtol=1e-6, atol=1e-7)
    assert optimizer._steps == [3, 2]
    assert optimizer.max_staging_bytes == 5 * 1024 * 1024


def test_chunked_adamw_checkpoint_resume(tmp_path):
    initial = torch.linspace(-1, 1, 257, dtype=torch.float32)
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = NVMeChunkedAdamW(
        [("weight", parameter)],
        path=str(tmp_path / "first"),
        chunk_size_mb=1,
        lr=1e-3,
    )
    parameter.grad = torch.linspace(1, -1, 257, dtype=torch.float32)
    optimizer.step()
    optimizer.param_groups[0]["lr"] = 2e-3

    checkpoint_path = tmp_path / "checkpoint" / "optimizer.pt"
    optimizer.save_nvme_checkpoint(checkpoint_path)

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = NVMeChunkedAdamW(
        [("weight", resumed_parameter)],
        path=str(tmp_path / "second"),
        chunk_size_mb=1,
        lr=9e-3,
    )
    resumed_optimizer.load_nvme_checkpoint(checkpoint_path)
    assert resumed_optimizer.param_groups[0]["lr"] == 2e-3
    assert resumed_optimizer._steps == [1]

    gradient = torch.sin(torch.arange(257, dtype=torch.float32))
    parameter.grad = gradient.clone()
    resumed_parameter.grad = gradient.clone()
    optimizer.step()
    resumed_optimizer.step()

    assert torch.equal(resumed_parameter, parameter)
    assert resumed_optimizer._steps == optimizer._steps == [2]


def test_bf16_parameters_use_fp32_master_weights(tmp_path):
    initial = torch.linspace(-2, 2, 1025, dtype=torch.bfloat16)
    parameter = torch.nn.Parameter(initial.clone())
    reference_parameter = torch.nn.Parameter(initial.float())
    optimizer = NVMeChunkedAdamW(
        [("weight", parameter)],
        path=str(tmp_path / "nvme"),
        chunk_size_mb=1,
        lr=2e-3,
        weight_decay=0.03,
    )
    reference = torch.optim.AdamW(
        [reference_parameter],
        lr=2e-3,
        weight_decay=0.03,
        foreach=False,
    )

    for step in range(3):
        gradient = torch.cos(torch.arange(1025, dtype=torch.float32) + step).to(torch.bfloat16)
        parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.float()
        optimizer.step()
        reference.step()
        optimizer.zero_grad()
        reference.zero_grad()

    assert torch.equal(parameter, reference_parameter.detach().to(torch.bfloat16))


def test_optimizer_can_keep_gradients_and_skip_master_file(tmp_path):
    initial = torch.linspace(-1, 1, 65)
    parameter = torch.nn.Parameter(initial.clone())
    reference_parameter = torch.nn.Parameter(initial.clone())
    optimizer = NVMeChunkedAdamW(
        [("weight", parameter)],
        path=str(tmp_path / "nvme"),
        chunk_size_mb=1,
        offload_gradients=False,
        master_weights=False,
        lr=1e-3,
    )
    reference = torch.optim.AdamW([reference_parameter], lr=1e-3, foreach=False)
    gradient = torch.sin(torch.arange(65, dtype=torch.float32))
    parameter.grad = gradient.clone()
    reference_parameter.grad = gradient.clone()

    optimizer.step()
    reference.step()

    assert torch.equal(parameter, reference_parameter)
    assert parameter.grad is not None
    assert optimizer.estimated_nvme_bytes == 2 * parameter.numel() * torch.float32.itemsize


def test_reset_nvme_state_reinitializes_master_from_parameter(tmp_path):
    parameter = torch.nn.Parameter(torch.arange(32, dtype=torch.float32))
    optimizer = NVMeChunkedAdamW(
        [("weight", parameter)],
        path=str(tmp_path / "first"),
        chunk_size_mb=1,
        lr=1e-3,
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    with torch.no_grad():
        parameter.copy_(torch.linspace(-1, 1, 32))
    optimizer.reset_nvme_state()
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    reference = torch.optim.AdamW([reference_parameter], lr=1e-3, foreach=False)
    gradient = torch.linspace(1, -1, 32)
    parameter.grad = gradient.clone()
    reference_parameter.grad = gradient.clone()
    optimizer.step()
    reference.step()

    assert torch.equal(parameter, reference_parameter)
    assert optimizer._steps == [1]
