import numpy as np
import torch
from torch import nn


class QuantileEmbedding(nn.Module):
    def __init__(self, state_dims, embedding_dims=64, norm_factory=nn.BatchNorm1d):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.hidden = nn.Sequential(
            nn.Linear(self.embedding_dims, self.embedding_dims),
            nn.LeakyReLU(0.2),
            norm_factory(self.embedding_dims),
        )
        self.to_state = nn.Sequential(
            nn.Linear(self.embedding_dims, state_dims),
            norm_factory(state_dims),
        )

    def forward(self, quantiles):
        quantiles = quantiles.repeat(1, self.embedding_dims)
        qs = self.hidden(quantiles)
        return self.to_state(qs)


class WeightedQuantileEmbedding(nn.Module):
    def __init__(self, state_dims, embedding_dims=20, **_):
        super().__init__()
        self.quantile_embeddings = nn.Embedding(
            embedding_dims, state_dims
        )
        self.quantile_indexes = nn.Parameter(
            torch.linspace(0, 1, embedding_dims), requires_grad=False
        )
        self.embedding_dims = embedding_dims

    def forward(self, quantiles):
        """quantiles.shape == (batch, 1)"""
        quantile_weights = (quantiles - self.quantile_indexes).abs()
        quantile_weights = 1. / (quantile_weights + 1e-8)
        quantile_weights = quantile_weights / quantile_weights.sum(-1, keepdim=True)
        weighted_embeddings = torch.mm(
            quantile_weights,
            self.quantile_embeddings.weight
        )
        return weighted_embeddings


class IQN(nn.Module):
    def __init__(
        self, feature_dims, quantile_dims=20, num_quantiles=8, mix='mult',
        quantile_embedding_factory=WeightedQuantileEmbedding,
        norm_factory=nn.BatchNorm1d#nn.Identity#
    ):
        super().__init__()
        self.quantile_embedding = quantile_embedding_factory(
            feature_dims, quantile_dims, norm_factory=norm_factory
        )
        self.feature_dims = feature_dims
        self.num_quantiles = num_quantiles
        self.mix = mix
        self._device = None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.repeat(self.num_quantiles, 1)  # (batch * qs, feature_dims)
        quantiles = self.sample_quantiles(batch_size)  # (batch * qs, 1)
        quantile_embedding = self.quantile_embedding(quantiles)  # (batch * qs, feature_dims)
        # if np.random.uniform() < 0.01:
        #     print(quantile_embedding)
        if self.mix == 'add':
            return x + quantile_embedding, quantiles
        elif self.mix.startswith('mult'):
            return x * quantile_embedding, quantiles
        else:
            raise ValueError(f'Unknown mix method {self.mix}')

    def sample_quantiles(self, n=1):
        if self._device is None:
            self._device = next(self.parameters()).device
        return torch.rand(n * self.num_quantiles, 1).to(self._device)


def iqn_loss(preds, target, taus, k=1.):
    assert not target.requires_grad
    batch_size = target.shape[0]
    if len(target.shape) == 1:
        output_dims = 1
        target = target[..., None]
    else:
        output_dims = target.shape[-1]
    num_quantiles = preds.shape[0] // batch_size
    taus = torch.reshape(taus, (-1, batch_size, output_dims))
    preds = torch.reshape(preds, (-1, batch_size, output_dims))
    target = target.repeat(num_quantiles, 1)
    target = torch.reshape(target, (-1, batch_size, output_dims))
    err = target - preds
    loss = torch.where(
        torch.abs(err) <= k,
        0.5 * err.pow(2),
        k * (torch.abs(err) - 0.5 * k)
    )
    return (torch.abs(taus - (err < 0).float()) * loss).sum(-1).mean()
