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

"""Synchronous, chunked NVMe optimizer state for FSDP2 parameter shards."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

_STORE_FORMAT = "verl-synchronous-nvme-tensor-store"
_STORE_VERSION = 1
_OPTIMIZER_FORMAT = "verl-nvme-chunked-adamw"
_OPTIMIZER_VERSION = 1
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_COPY_BUFFER_BYTES = 8 * 1024 * 1024


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported tensor dtype in NVMe manifest: {name}")
    return dtype


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _atomic_write_json(path: Path, value: dict[str, Any], fsync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        if fsync:
            os.fsync(output.fileno())
    os.replace(temporary_path, path)


def _copy_exact(source: Path, destination: Path, expected_size: int, fsync: bool = False) -> None:
    actual_size = source.stat().st_size
    if actual_size != expected_size:
        raise OSError(f"NVMe tensor file {source} has {actual_size} bytes; expected {expected_size}")

    buffer = bytearray(min(_COPY_BUFFER_BYTES, max(expected_size, 1)))
    with source.open("rb", buffering=0) as input_file, destination.open("xb", buffering=0) as output_file:
        while True:
            count = input_file.readinto(buffer)
            if count == 0:
                break
            view = memoryview(buffer)[:count]
            written = 0
            while written < count:
                result = output_file.write(view[written:])
                if not result:
                    raise OSError(f"Short write while copying NVMe tensor file to {destination}")
                written += result
        if fsync:
            os.fsync(output_file.fileno())


class SynchronousNVMeStore:
    """Raw tensor files accessed with blocking ``readinto`` and ``write`` calls."""

    def __init__(self, root: str | os.PathLike[str], *, fsync: bool = False):
        self.root = Path(root)
        self.fsync = fsync
        self._records: dict[str, dict[str, Any]] = {}
        self.root.mkdir(parents=True, exist_ok=False)

    @property
    def allocated_bytes(self) -> int:
        return sum(record["nbytes"] for record in self._records.values())

    def register(self, key: str, *, numel: int, dtype: torch.dtype) -> None:
        if not _SAFE_KEY.fullmatch(key):
            raise ValueError(f"Unsafe NVMe tensor key: {key!r}")
        if key in self._records:
            raise ValueError(f"NVMe tensor key is already registered: {key}")
        if numel < 0:
            raise ValueError("NVMe tensor numel must be non-negative")

        filename = f"{key}.bin"
        nbytes = numel * dtype.itemsize
        path = self.root / filename
        with path.open("xb") as output:
            output.truncate(nbytes)
        self._records[key] = {
            "dtype": _dtype_name(dtype),
            "filename": filename,
            "nbytes": nbytes,
            "numel": numel,
        }

    def write_manifest(self, metadata: dict[str, Any] | None = None) -> None:
        manifest = {
            "format": _STORE_FORMAT,
            "version": _STORE_VERSION,
            "tensors": self._records,
            "metadata": metadata or {},
        }
        _atomic_write_json(self.root / "manifest.json", manifest, fsync=self.fsync)

    def read(self, key: str, *, offset: int, count: int) -> torch.Tensor:
        record = self._validate_range(key, offset, count)
        path = self.root / record["filename"]
        self._validate_file_size(path, record["nbytes"])
        dtype = _dtype_from_name(record["dtype"])
        tensor = torch.empty(count, dtype=dtype, device="cpu")
        expected_bytes = count * dtype.itemsize
        if expected_bytes == 0:
            return tensor

        byte_view = memoryview(tensor.view(torch.uint8).numpy())
        bytes_read = 0
        with path.open("rb", buffering=0) as input_file:
            input_file.seek(offset * dtype.itemsize)
            while bytes_read < expected_bytes:
                result = input_file.readinto(byte_view[bytes_read:])
                if not result:
                    raise OSError(f"Short read from NVMe tensor file {path}")
                bytes_read += result
        return tensor

    def write(self, key: str, tensor: torch.Tensor, *, offset: int) -> None:
        record = self._records.get(key)
        if record is None:
            raise KeyError(f"Unknown NVMe tensor key: {key}")
        dtype = _dtype_from_name(record["dtype"])
        cpu_tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous().view(-1)
        self._validate_range(key, offset, cpu_tensor.numel())
        path = self.root / record["filename"]
        self._validate_file_size(path, record["nbytes"])
        byte_count = cpu_tensor.numel() * dtype.itemsize
        if byte_count == 0:
            return

        byte_view = memoryview(cpu_tensor.view(torch.uint8).numpy())
        bytes_written = 0
        with path.open("r+b", buffering=0) as output_file:
            output_file.seek(offset * dtype.itemsize)
            while bytes_written < byte_count:
                result = output_file.write(byte_view[bytes_written:])
                if not result:
                    raise OSError(f"Short write to NVMe tensor file {path}")
                bytes_written += result
            if self.fsync:
                os.fsync(output_file.fileno())

    def reset(self, key: str) -> None:
        record = self._records.get(key)
        if record is None:
            raise KeyError(f"Unknown NVMe tensor key: {key}")
        path = self.root / record["filename"]
        with path.open("wb") as output:
            output.truncate(record["nbytes"])
            if self.fsync:
                output.flush()
                os.fsync(output.fileno())

    def snapshot(self, destination: str | os.PathLike[str], keys: Iterable[str]) -> None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        snapshot_records: dict[str, dict[str, Any]] = {}
        for key in keys:
            record = self._records[key]
            _copy_exact(
                self.root / record["filename"],
                destination / record["filename"],
                record["nbytes"],
                fsync=self.fsync,
            )
            snapshot_records[key] = record
        _atomic_write_json(
            destination / "manifest.json",
            {"format": _STORE_FORMAT, "version": _STORE_VERSION, "tensors": snapshot_records},
            fsync=self.fsync,
        )

    def restore(self, source: str | os.PathLike[str], keys: Iterable[str]) -> None:
        source = Path(source)
        with (source / "manifest.json").open(encoding="utf-8") as input_file:
            manifest = json.load(input_file)
        if manifest.get("format") != _STORE_FORMAT or manifest.get("version") != _STORE_VERSION:
            raise ValueError(f"Unsupported NVMe tensor-store manifest in {source}")

        source_records = manifest.get("tensors", {})
        keys = list(keys)
        for key in keys:
            if source_records.get(key) != self._records.get(key):
                raise ValueError(f"NVMe checkpoint tensor metadata does not match for {key}")
            record = self._records[key]
            self._validate_file_size(source / record["filename"], record["nbytes"])

        for key in keys:
            record = self._records[key]
            destination = self.root / record["filename"]
            temporary_path = destination.with_name(f".{destination.name}.restore-{os.getpid()}")
            if temporary_path.exists():
                temporary_path.unlink()
            _copy_exact(source / record["filename"], temporary_path, record["nbytes"], fsync=self.fsync)
            os.replace(temporary_path, destination)

    def _validate_range(self, key: str, offset: int, count: int) -> dict[str, Any]:
        record = self._records.get(key)
        if record is None:
            raise KeyError(f"Unknown NVMe tensor key: {key}")
        if offset < 0 or count < 0 or offset + count > record["numel"]:
            raise ValueError(
                f"NVMe tensor range [{offset}, {offset + count}) is outside {key} with {record['numel']} elements"
            )
        return record

    @staticmethod
    def _validate_file_size(path: Path, expected_size: int) -> None:
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise OSError(f"NVMe tensor file {path} has {actual_size} bytes; expected {expected_size}")


class NVMeChunkedAdamW(Optimizer):
    """AdamW over local FSDP2 shards with moments and optional gradients on NVMe."""

    _verl_nvme_optimizer = True

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.Tensor]],
        *,
        path: str,
        chunk_size_mb: int = 256,
        state_dtype: str = "fp32",
        offload_gradients: bool = True,
        master_weights: bool = True,
        fsync: bool = False,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        named_parameters = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
        if not named_parameters:
            raise ValueError("NVMeChunkedAdamW requires at least one trainable parameter")
        self._validate_hyperparameters(lr, betas, eps, weight_decay)
        if state_dtype.lower() not in {"fp32", "float32", "torch.float32"}:
            raise ValueError("NVMeChunkedAdamW currently supports only FP32 optimizer state")
        if chunk_size_mb <= 0:
            raise ValueError("chunk_size_mb must be greater than zero")

        defaults = {"lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay}
        super().__init__([parameter for _, parameter in named_parameters], defaults)

        self._names = [name for name, _ in named_parameters]
        if len(self._names) != len(set(self._names)):
            raise ValueError("NVMeChunkedAdamW parameter names must be unique")
        self._parameters = [parameter for _, parameter in named_parameters]
        self._parameter_index = {id(parameter): index for index, parameter in enumerate(self._parameters)}
        self._steps = [0] * len(self._parameters)
        self._state_dtype = torch.float32
        self._chunk_size_bytes = chunk_size_mb * 1024 * 1024
        self._chunk_numel = max(1, self._chunk_size_bytes // self._state_dtype.itemsize)
        self._offload_gradients = offload_gradients
        self._master_weights = master_weights
        self._fsync = fsync
        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        run_id = os.environ.get("VERL_NVME_RUN_ID", f"{time.time_ns()}-{os.getpid()}")
        store_root = Path(path).expanduser() / f"run_{run_id}" / f"rank_{self._rank:05d}"
        self.store = SynchronousNVMeStore(store_root, fsync=fsync)
        self._parameter_metadata = self._register_state_files()
        self._validate_capacity()
        self._initialize_master_weights()
        self.store.write_manifest(
            {
                "optimizer_format": _OPTIMIZER_FORMAT,
                "parameter_metadata": self._parameter_metadata,
                "rank": self._rank,
                "world_size": self._world_size,
            }
        )
        logger.info(
            "Initialized rank %d NVMe optimizer store at %s (logical size %.2f GiB)",
            self._rank,
            self.store.root,
            self.store.allocated_bytes / 1024**3,
        )

    @property
    def estimated_nvme_bytes(self) -> int:
        return self.store.allocated_bytes

    @property
    def max_staging_bytes(self) -> int:
        # gradient, master parameter, two moments, and Adam denominator
        return 5 * self._chunk_size_bytes

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active_indices = self._spill_gradients() if self._offload_gradients else self._active_parameter_indices()
        active_indices = set(active_indices)
        for group in self.param_groups:
            for parameter in group["params"]:
                index = self._parameter_index[id(parameter)]
                if index not in active_indices:
                    continue
                self._update_parameter(index, parameter, group)
        return loss

    def state_dict(self):
        raise RuntimeError(
            "NVMeChunkedAdamW state cannot be represented by optimizer.state_dict(); use save_nvme_checkpoint()"
        )

    def load_state_dict(self, state_dict):
        raise RuntimeError(
            "NVMeChunkedAdamW state cannot be restored by optimizer.load_state_dict(); use load_nvme_checkpoint()"
        )

    def save_nvme_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state_directory = checkpoint_path.with_name(f"{checkpoint_path.name}.nvme")
        temporary_directory = state_directory.with_name(f".{state_directory.name}.tmp-{os.getpid()}")
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        self.store.snapshot(temporary_directory, self._persistent_keys())
        if state_directory.exists():
            shutil.rmtree(state_directory)
        os.replace(temporary_directory, state_directory)

        descriptor = {
            "format": _OPTIMIZER_FORMAT,
            "version": _OPTIMIZER_VERSION,
            "rank": self._rank,
            "world_size": self._world_size,
            "state_directory": state_directory.name,
            "parameter_metadata": self._parameter_metadata,
            "steps": self._steps,
            "param_groups": self._serialize_param_groups(),
            "master_weights": self._master_weights,
            "offload_gradients": self._offload_gradients,
        }
        _atomic_write_json(checkpoint_path, descriptor, fsync=self._fsync)

    def load_nvme_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        checkpoint_path = Path(checkpoint_path)
        with checkpoint_path.open(encoding="utf-8") as input_file:
            descriptor = json.load(input_file)
        if descriptor.get("format") != _OPTIMIZER_FORMAT or descriptor.get("version") != _OPTIMIZER_VERSION:
            raise ValueError(f"Unsupported NVMe optimizer checkpoint: {checkpoint_path}")
        if descriptor.get("rank") != self._rank or descriptor.get("world_size") != self._world_size:
            raise ValueError("NVMe optimizer checkpoint rank or world size does not match the current worker")
        if descriptor.get("parameter_metadata") != self._parameter_metadata:
            raise ValueError("NVMe optimizer checkpoint parameter layout does not match the current model")
        if descriptor.get("master_weights") != self._master_weights:
            raise ValueError("NVMe optimizer checkpoint master_weights setting does not match")

        directory_name = descriptor.get("state_directory", "")
        if directory_name != f"{checkpoint_path.name}.nvme":
            raise ValueError("NVMe optimizer checkpoint contains an unsafe state directory")
        self.store.restore(checkpoint_path.parent / directory_name, self._persistent_keys())

        steps = descriptor.get("steps")
        if not isinstance(steps, list) or len(steps) != len(self._steps) or any(not isinstance(x, int) for x in steps):
            raise ValueError("NVMe optimizer checkpoint has invalid per-parameter steps")
        self._steps = steps
        self._restore_param_groups(descriptor.get("param_groups"))

    @torch.no_grad()
    def reset_nvme_state(self) -> None:
        """Reset moments after loading a model-only checkpoint."""
        self._steps = [0] * len(self._steps)
        for index in range(len(self._parameters)):
            self.store.reset(self._key(index, "exp_avg"))
            self.store.reset(self._key(index, "exp_avg_sq"))
            if self._offload_gradients:
                self.store.reset(self._key(index, "grad"))
            if self._master_weights:
                self.store.reset(self._key(index, "master"))
        self._initialize_master_weights()

    def _register_state_files(self) -> list[dict[str, Any]]:
        metadata = []
        for index, (name, parameter) in enumerate(zip(self._names, self._parameters, strict=True)):
            local_parameter = _local_tensor(parameter)
            if not local_parameter.is_floating_point() or local_parameter.is_complex():
                raise TypeError(f"NVMeChunkedAdamW does not support parameter dtype {local_parameter.dtype}: {name}")
            if not local_parameter.is_contiguous():
                raise ValueError(f"NVMeChunkedAdamW requires contiguous local parameter shards: {name}")
            numel = local_parameter.numel()
            self.store.register(self._key(index, "exp_avg"), numel=numel, dtype=self._state_dtype)
            self.store.register(self._key(index, "exp_avg_sq"), numel=numel, dtype=self._state_dtype)
            if self._offload_gradients:
                self.store.register(self._key(index, "grad"), numel=numel, dtype=self._state_dtype)
            if self._master_weights:
                self.store.register(self._key(index, "master"), numel=numel, dtype=self._state_dtype)
            metadata.append(
                {
                    "name": name,
                    "global_shape": list(parameter.shape),
                    "local_shape": list(local_parameter.shape),
                    "numel": numel,
                    "parameter_dtype": _dtype_name(local_parameter.dtype),
                }
            )
        return metadata

    def _initialize_master_weights(self) -> None:
        if not self._master_weights:
            return
        for index, parameter in enumerate(self._parameters):
            local_parameter = _local_tensor(parameter).detach().view(-1)
            for offset, count in self._ranges(local_parameter.numel()):
                self.store.write(self._key(index, "master"), local_parameter[offset : offset + count], offset=offset)

    def _validate_capacity(self) -> None:
        available_bytes = shutil.disk_usage(self.store.root).free
        required_bytes = self.store.allocated_bytes
        if required_bytes > available_bytes:
            raise OSError(
                f"Rank {self._rank} NVMe optimizer state requires {required_bytes / 1024**3:.2f} GiB, "
                f"but only {available_bytes / 1024**3:.2f} GiB is available at {self.store.root}"
            )

    def _active_parameter_indices(self) -> list[int]:
        return [index for index, parameter in enumerate(self._parameters) if parameter.grad is not None]

    def _spill_gradients(self) -> list[int]:
        active_indices = []
        for index, parameter in enumerate(self._parameters):
            if parameter.grad is None:
                continue
            local_gradient = self._validated_local_gradient(index, parameter)
            for offset, count in self._ranges(local_gradient.numel()):
                self.store.write(self._key(index, "grad"), local_gradient[offset : offset + count], offset=offset)
            parameter.grad = None
            active_indices.append(index)
        return active_indices

    def _update_parameter(self, index: int, parameter: torch.Tensor, group: dict[str, Any]) -> None:
        local_parameter = _local_tensor(parameter).view(-1)
        local_gradient = None if self._offload_gradients else self._validated_local_gradient(index, parameter)
        step = self._steps[index] + 1
        beta1, beta2 = group["betas"]
        learning_rate = group["lr"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]
        bias_correction1 = 1 - beta1**step
        bias_correction2_sqrt = math.sqrt(1 - beta2**step)

        for offset, count in self._ranges(local_parameter.numel()):
            if self._offload_gradients:
                gradient = self.store.read(self._key(index, "grad"), offset=offset, count=count)
            else:
                gradient = local_gradient[offset : offset + count].to(device="cpu", dtype=self._state_dtype)
            exp_avg = self.store.read(self._key(index, "exp_avg"), offset=offset, count=count)
            exp_avg_sq = self.store.read(self._key(index, "exp_avg_sq"), offset=offset, count=count)
            if self._master_weights:
                master_parameter = self.store.read(self._key(index, "master"), offset=offset, count=count)
            else:
                master_parameter = local_parameter[offset : offset + count].to(device="cpu", dtype=self._state_dtype)

            master_parameter.mul_(1 - learning_rate * weight_decay)
            exp_avg.lerp_(gradient, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            denominator = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
            master_parameter.addcdiv_(exp_avg, denominator, value=-learning_rate / bias_correction1)

            self.store.write(self._key(index, "exp_avg"), exp_avg, offset=offset)
            self.store.write(self._key(index, "exp_avg_sq"), exp_avg_sq, offset=offset)
            if self._master_weights:
                self.store.write(self._key(index, "master"), master_parameter, offset=offset)
            local_parameter[offset : offset + count].copy_(
                master_parameter.to(device=local_parameter.device, dtype=local_parameter.dtype)
            )
        self._steps[index] = step

    def _validated_local_gradient(self, index: int, parameter: torch.Tensor) -> torch.Tensor:
        local_gradient = _local_tensor(parameter.grad).detach()
        if local_gradient.is_sparse:
            raise TypeError("NVMeChunkedAdamW does not support sparse gradients")
        if not local_gradient.is_contiguous():
            raise ValueError(f"NVMeChunkedAdamW requires contiguous local gradients: {self._names[index]}")
        local_gradient = local_gradient.view(-1)
        if local_gradient.numel() != self._parameter_metadata[index]["numel"]:
            raise ValueError(f"Gradient shape does not match local parameter shard: {self._names[index]}")
        return local_gradient

    def _persistent_keys(self) -> list[str]:
        keys = []
        for index in range(len(self._parameters)):
            keys.extend((self._key(index, "exp_avg"), self._key(index, "exp_avg_sq")))
            if self._master_weights:
                keys.append(self._key(index, "master"))
        return keys

    def _serialize_param_groups(self) -> list[dict[str, Any]]:
        groups = []
        for group in self.param_groups:
            serialized = {key: value for key, value in group.items() if key != "params"}
            serialized["betas"] = list(serialized["betas"])
            serialized["params"] = [self._parameter_index[id(parameter)] for parameter in group["params"]]
            groups.append(serialized)
        return groups

    def _restore_param_groups(self, saved_groups: Any) -> None:
        if not isinstance(saved_groups, list) or len(saved_groups) != len(self.param_groups):
            raise ValueError("NVMe optimizer checkpoint parameter groups do not match")
        for current, saved in zip(self.param_groups, saved_groups, strict=True):
            expected_parameters = [self._parameter_index[id(parameter)] for parameter in current["params"]]
            if saved.get("params") != expected_parameters:
                raise ValueError("NVMe optimizer checkpoint parameter-group layout does not match")
            parameters = current["params"]
            current.clear()
            current.update(saved)
            current["params"] = parameters
            current["betas"] = tuple(current["betas"])

    def _ranges(self, numel: int):
        for offset in range(0, numel, self._chunk_numel):
            yield offset, min(self._chunk_numel, numel - offset)

    @staticmethod
    def _key(index: int, state: str) -> str:
        return f"param_{index:06d}.{state}"

    @staticmethod
    def _validate_hyperparameters(lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 2 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
