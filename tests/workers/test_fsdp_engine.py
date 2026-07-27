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

import torch

from verl.workers.engine.fsdp.transformer_impl import _scale_logits_by_temperature


class _CustomAutogradIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def test_temperature_scaling_accepts_custom_autograd_view():
    base_logits = torch.randn(1, 4, 8, requires_grad=True)
    logits_view = _CustomAutogradIdentity.apply(base_logits).squeeze(0)
    temperatures = torch.full((4, 1), 0.5)

    scaled_logits = _scale_logits_by_temperature(logits_view, temperatures)
    scaled_logits.sum().backward()

    assert torch.equal(base_logits.grad, torch.full_like(base_logits, 2.0))
