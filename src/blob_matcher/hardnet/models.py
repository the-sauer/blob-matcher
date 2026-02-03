# Copyright 2019 EPFL, Google LLC
# Copyright 2025 Hendrik Sauer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class HardNet(nn.Module):
    def __init__(self, patch_size=32):
        super(HardNet, self).__init__()
        self.patch_size = patch_size

        # model processing patches of size [32 x 32] and giving description vectors of length 2**7
        if patch_size == 32:
            kernel_size = 3
            padding = 1
            pool = 8
        elif patch_size == 64:
            kernel_size = 5
            padding = 2
            pool = 16
        elif patch_size == 128:
            kernel_size = 9
            padding = 4
            pool = 32
        else:
            raise ValueError(f"Unsupported patch size {patch_size}")

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding, bias=False),             # 32x32
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding, bias=False),            # 32x32
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2, padding=padding, bias=False),  # 16x16
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, bias=False),            # 16x16
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padding, bias=False), # 8x8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding, bias=False),          # 8x8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=(pool, 1)),
            nn.Conv2d(128, 128, (1, pool), bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        # initialize weights
        self.features.apply(weights_init)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return ((x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
            sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))

    # function to forward-propagate inputs through the network
    def forward(self, patches):
        x_features = self.features(self.input_norm(patches))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x), patches


def weights_init(m):
    '''
    Conv2d module weight initialization method
    '''

    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return
