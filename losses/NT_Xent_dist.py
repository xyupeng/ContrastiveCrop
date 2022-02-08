# Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import diffdist


def diff_gather(z):
    '''
        Return: gather list of z
    '''
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    # gather_z = torch.cat(gather_z, dim=0)
    return gather_z


class NT_Xent_dist(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(NT_Xent_dist, self).__init__()
        self.temperature = temperature
        self.gather_op = diff_gather
        self.world_size = dist.get_world_size()
        self.base_temperature = base_temperature

    def forward(self, feat1, feat2):
        """
        implement based on pos_mask & neg_mask; could also use torch.diag & nn.CrossEntropyLoss
        Args:
            feat1, feat2: feats of view1, view2; feat1.shape == feat2.shape == (batch_size, C)
        Returns:
            A loss scalar.
        """

        bsz_gpu = feat1.shape[0]
        N = bsz_gpu * self.world_size  # total batch_size

        # compute logits
        feat1 = torch.cat(self.gather_op(feat1))
        feat2 = torch.cat(self.gather_op(feat2))
        features = torch.cat([feat1, feat2], dim=0)
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # neg_mask: denominator; mask-out self-contrast cases
        neg_mask = ~torch.eye(2 * N, dtype=torch.bool).cuda()
        # pos_mask: numerator; single positive pair
        pos_mask = torch.zeros((2 * N, 2 * N), dtype=torch.bool).cuda()
        pos_mask[:N, N:] = torch.eye(N)
        pos_mask[N:, :N] = torch.eye(N)

        # compute log_prob
        exp_logits = torch.exp(logits)[neg_mask].view(2 * N, -1)  # on different gpus
        log_prob = logits[pos_mask] - torch.log(exp_logits.sum(1))

        # loss
        loss = -(self.temperature / self.base_temperature) * log_prob.mean()
        return loss
