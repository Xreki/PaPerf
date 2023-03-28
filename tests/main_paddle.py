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

import paddle
import paddle.nn as nn
from paperf import profile_paddle


class SimpleNet(nn.Layer):
    def __init__(self, in_features, out_features):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.softmax(out, axis=-1)
        return out


def main():
    batch_size = 64
    in_features = 16
    out_features = 32

    model = SimpleNet(in_features, out_features)
    adam = paddle.optimizer.Adam(weight_decay=0.01, parameters=model.parameters())

    profile_paddle.register_profile_hook(model)

    model.train()

    max_iters = 20
    for iter_id in range(max_iters):
        profile_paddle.switch_profile(iter_id, 10, 20)

        with profile_paddle.add_nvtx_event("forward"):
            x = paddle.randn(shape=[batch_size, in_features])
            out = model(x)
            loss = paddle.mean(out)
        with profile_paddle.add_nvtx_event("backward"):
            loss.backward()
        with profile_paddle.add_nvtx_event("optimizer"):
            adam.step()
        with profile_paddle.add_nvtx_event("clear_grad"):
            adam.clear_grad()


main()
