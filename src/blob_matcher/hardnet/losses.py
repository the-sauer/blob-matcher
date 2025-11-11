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
import sys


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) +
                       torch.t(d2_sq.repeat(1, anchor.size(0))) - 2.0 *
                       torch.bmm(anchor.unsqueeze(0),
                                 torch.t(positive).unsqueeze(0)).squeeze(0)) +
                      eps)


def loss_HardNet_weighted(anchor,
                          positive,
                          garbage,
                          anchor_swap=False,
                          margin=1.0,
                          batch_reduce='min',
                          loss_type="triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(
    ), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).to(anchor.device)

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask
    if garbage is not None:
        garbage_dist_matrix = torch.concat((
            distance_matrix_vector(anchor, garbage),
            distance_matrix_vector(positive, garbage)
        ))
    if batch_reduce == 'min':
        min_neg_idx = torch.min(dist_without_min_on_diag, 1)[1]

        min_neg = torch.min(dist_without_min_on_diag, 1)[0]
        if garbage is not None:
            min_neg_garbage = torch.min(garbage_dist_matrix, 1)[0]
            assert min_neg_garbage.size(0) == 2 * min_neg.size(0)
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag, 0)[0]
            min_neg = torch.min(
                min_neg,
                min_neg2
            )
            if garbage is not None:
                min_neg = torch.min(
                    min_neg,
                    min_neg_garbage[:min_neg.size(0)]
                )
                min_neg = torch.min(
                    min_neg,
                    min_neg_garbage[min_neg.size(0):]
                )

        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        raise NotImplementedError("batch reduce: average is not yet implemented for this loss function")
        pos = pos1.repeat(anchor.size(0)).view(-1, 1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1, 1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(
                -1, 1)
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        raise NotImplementedError("batch reduce: random is not yet implemented for this loss function")
        idxs = torch.autograd.Variable(
            torch.randperm(anchor.size()[0]).long()).device(anchor.device)
        min_neg = dist_without_min_on_diag.gather(1, idxs.view(-1, 1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(
                1, idxs.view(-1, 1))
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else:
        raise ValueError(f'Unknown batch reduce mode "{batch_reduce}". Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        raise NotImplementedError("loss type: softmax is not yet implemented for this loss function")
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = -torch.log(exp_pos / exp_den)
    elif loss_type == 'contrastive':
        raise NotImplementedError("loss type: contrastive is not yet implemented for this loss function")
        loss = torch.clamp(margin - min_neg, min=0.0) + pos
    else:
        raise ValueError(f'Unknown loss type "{loss}". Try triplet_margin, softmax or contrastive')
    loss = torch.mean(loss)
    return loss, min_neg_idx


# For testing and debugging
if __name__ == "__main__":
    anchor = torch.rand(10, 128)#.cuda()
    positive = torch.rand(10, 128)#.cuda() 
    garbage = torch.rand(0, 128)#.cuda()

    loss, _ = loss_HardNet_weighted(anchor, positive, garbage, anchor_swap=True)
    print(loss)
