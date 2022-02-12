import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOL(nn.Module):
    """
    Build a BYOL model. https://arxiv.org/abs/2006.07733
    """
    def __init__(self, encoder_q, encoder_k, dim=4096, pred_dim=256, m=0.996):
        """
        encoder_q: online network
        encoder_k: target network
        dim: feature dimension (default: 4096)
        pred_dim: hidden dimension of the predictor (default: 256)
        """
        super(BYOL, self).__init__()

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.m = m

        # projector
        # encoder_dim = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                          # nn.Linear(encoder_dim, dim, bias=False),
                                          nn.BatchNorm1d(dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(dim, pred_dim))
        self.encoder_k.fc = nn.Sequential(self.encoder_k.fc,
                                          # nn.Linear(encoder_dim, dim, bias=False),
                                          nn.BatchNorm1d(dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(dim, pred_dim))

        self.predictor = nn.Sequential(nn.Linear(pred_dim, dim),
                                       nn.BatchNorm1d(dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(dim, pred_dim))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        """

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        p1 = self.predictor(self.encoder_q(x1))  # NxC
        z2 = self.encoder_k(x2)  # NxC

        p2 = self.predictor(self.encoder_q(x2))  # NxC
        z1 = self.encoder_k(x1)  # NxC

        return p1, p2, z1.detach(), z2.detach()
