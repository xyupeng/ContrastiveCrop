# Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class NT_Xent(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, feat1, feat2):
        """
        implement based on pos_mask & neg_mask; could also use torch.diag & nn.CrossEntropyLoss
        Args:
            feat1, feat2: feats of view1, view2; feat1.shape == feat2.shape == (batch_size, C)
        Returns:
            A loss scalar.
        """
        # works for DataParallel; default cuda:0
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        batch_size = feat1.shape[0]
        # compute logits
        features = torch.cat([feat1, feat2], dim=0)
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # neg_mask: denominator; mask out self-contrast cases
        neg_mask = ~torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        '''neg_mask, batch_size=4
                    |
            0 1 1 1 | 1 1 1 1
            1 0 1 1 | 1 1 1 1
            1 1 0 1 | 1 1 1 1
            1 1 1 0 | 1 1 1 1
          ----------|---------- 
            1 1 1 1 | 0 1 1 1
            1 1 1 1 | 1 0 1 1
            1 1 1 1 | 1 1 0 1
            1 1 1 1 | 1 1 1 0
                    |
        '''
        # pos_mask: numerator; single positive pair
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool).to(device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
        '''pos_mask, batch_size=4
                    |
            0 0 0 0 | 1 0 0 0
            0 0 0 0 | 0 1 0 0
            0 0 0 0 | 0 0 1 0
            0 0 0 0 | 0 0 0 1
          ----------|---------- 
            1 0 0 0 | 0 0 0 0
            0 1 0 0 | 0 0 0 0
            0 0 1 0 | 0 0 0 0
            0 0 0 1 | 0 0 0 0
                    |
        '''

        # compute log_prob
        exp_logits = torch.exp(logits)[neg_mask].view(2 * batch_size, -1)
        log_prob = logits[pos_mask] - torch.log(exp_logits.sum(1))

        # loss
        loss = -(self.temperature / self.base_temperature) * log_prob.mean()
        return loss
