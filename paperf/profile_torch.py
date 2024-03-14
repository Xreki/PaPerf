# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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
import contextlib


_PROFILER_ENABLED = False


def _forward_pre_hook(layer, inputs):
    # print(f"[{layer.__class__.__name__}]")
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(layer.__class__.__name__ + "_fwd")
    return None


def _forward_post_hook(module, inputs, outputs):
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_pop()


def _register_hook_recursively(module, pre_hook, post_hook):
    if not isinstance(module, torch.nn.Module):
        return

    for submodule in module.children():
        _register_hook_recursively(submodule, pre_hook, post_hook)

    if pre_hook is not None:
        module.register_forward_pre_hook(hook=pre_hook)
    if post_hook is not None:
        module.register_forward_hook(hook=post_hook)


def register_profile_hook(model):
    if isinstance(model, torch.nn.Module):
        _register_hook_recursively(model, _forward_pre_hook, _forward_post_hook)
    elif isinstance(model, list):
        for module in model:
            _register_hook_recursively(module, _forward_pre_hook, _forward_post_hook)


def switch_profile(iter_id, start, end, event_name=None):
    global _PROFILER_ENABLED
    if event_name is None:
        event_name = "iter_{}".format(iter_id)
    if iter_id == start:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        _PROFILER_ENABLED = True
        torch.cuda.nvtx.range_push(event_name)
    elif iter_id == end:
        torch.cuda.nvtx.range_pop()
        _PROFILER_ENABLED = False
        torch.cuda.cudart().cudaProfilerStop()
    elif iter_id > start and iter_id < end:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(event_name)


@contextlib.contextmanager
def add_record_event(event_name):
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(event_name)
        yield
        torch.cuda.nvtx.range_pop()
    else:
        yield


def push_record_event(event_name):
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(event_name)


def pop_record_event():
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_pop()
