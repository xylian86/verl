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
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.engine.fsdp.nvme_optimizer import NVMeChunkedAdamW


class TinyConfig:
    name_or_path = ""

    @staticmethod
    def save_pretrained(path):
        Path(path, "config.json").write_text("{}\n", encoding="utf-8")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(32, 64), torch.nn.Linear(64, 8)])
        self.config = TinyConfig()

    def forward(self, inputs):
        return self.layers[1](torch.nn.functional.gelu(self.layers[0](inputs)))

    @staticmethod
    def can_generate():
        return False


def _run_fsdp2_step(rank, world_size, rendezvous_path, nvme_path):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1234)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        model = TinyModel().to(device="cuda", dtype=torch.bfloat16)
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        optimizer = NVMeChunkedAdamW(
            model.named_parameters(),
            path=nvme_path,
            chunk_size_mb=1,
            lr=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
        checkpoint_manager = FSDPCheckpointManager(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            checkpoint_config=None,
        )
        before = [parameter.to_local().detach().clone() for parameter in model.parameters()]
        inputs = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
        loss = model(inputs).float().square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        assert all(parameter.grad is None for parameter in model.parameters())
        assert any(
            not torch.equal(previous, parameter.to_local())
            for previous, parameter in zip(before, model.parameters(), strict=True)
        )
        assert (optimizer.store.root / "manifest.json").is_file()

        saved_parameters = [parameter.to_local().detach().clone() for parameter in model.parameters()]
        checkpoint_path = str(Path(nvme_path).parent / "checkpoint")
        checkpoint_manager.save_checkpoint(checkpoint_path, global_step=1)

        model(torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)).float().square().mean().backward()
        optimizer.step()
        scheduler.step()
        checkpoint_manager.load_checkpoint(checkpoint_path)

        assert all(
            torch.equal(saved, parameter.to_local())
            for saved, parameter in zip(saved_parameters, model.parameters(), strict=True)
        )
        assert optimizer._steps == [1] * len(optimizer._steps)
        assert scheduler.last_epoch == 1
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_gpu_fsdp2_nvme_optimizer_step(tmp_path):
    rendezvous_fd, rendezvous_path = tempfile.mkstemp(dir=tmp_path)
    os.close(rendezvous_fd)
    Path(rendezvous_path).unlink()
    mp.spawn(
        _run_fsdp2_step,
        args=(2, rendezvous_path, str(tmp_path / "nvme")),
        nprocs=2,
        join=True,
    )
