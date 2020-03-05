import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention2d(nn.Module):
    """
    Follows https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """
    def __init__(self, in_dims, attention_dims=None):
        super().__init__()
        if attention_dims is None:
            attention_dims = in_dims // 8
        self.attention_dims = attention_dims
        self.query = nn.Sequential(
            nn.Conv2d(in_dims, attention_dims, 1, padding=0),
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_dims, attention_dims, 1, padding=0),
        )
        self.value = nn.Sequential(
            nn.Conv2d(in_dims, in_dims, 1, padding=0),
        )
        self.gamma = nn.Parameter(
            torch.zeros(1), requires_grad=True
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch, self.attention_dims, -1)
        k = k.view(batch, self.attention_dims, -1)
        v = v.view(batch, channels, -1)
        q = q.permute((0, 2, 1))
        attention = torch.matmul(q, k)
        attention = F.softmax(attention, dim=-1)
        attended = torch.matmul(v, attention.permute(0, 2, 1))
        attended = attended.view(batch, channels, height, width)
        return self.gamma * attended + x
