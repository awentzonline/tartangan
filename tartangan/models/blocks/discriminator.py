import functools

import torch
from torch import nn
import torch.nn.functional as F

from ..iqn import IQN, iqn_loss
from ..layers import Interpolate


class DiscriminatorInput(nn.Module):
    def __init__(self, in_dims, out_dims,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True),
            activation_factory(),
        )

    def forward(self, img):
        return self.convs(img)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_block=False,
                 norm_factory=nn.BatchNorm2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        layers = [
            norm_factory(out_dims),
            activation_factory(),
            nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            norm_factory(out_dims),
            activation_factory(),
            nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True),
            Interpolate(scale_factor=0.5, mode='bilinear', align_corners=True),
        ]
        if first_block:
            layers = layers[2:]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class ResidualDiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_block=False,
                 norm_factory=nn.BatchNorm2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        layers = [
            norm_factory(in_dims),
            activation_factory(),
            nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            norm_factory(out_dims),
            activation_factory(),
            nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True),
            Interpolate(scale_factor=0.5, mode='bilinear', align_corners=True),
        ]
        if first_block:
            layers = layers[2:]
        self.convs = nn.Sequential(*layers)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.project_input = None
        if in_dims != out_dims:
            # self.project_input = self._project_input
            self.project_input = nn.Sequential(
                nn.Conv2d(in_dims, out_dims, 1)
            )
        # map(nn.init.orthogonal_, self.parameters())

    def _project_input(self, x):
        new_shape = list(x.shape)
        new_shape[1] = self.out_dims
        zs = torch.zeros(*new_shape).to(x.device)
        zs[:, :self.in_dims] = x
        return zs

    def forward(self, x):
        h = self.convs(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        if self.project_input is not None:
            x = self.project_input(x)
        return x + h


class DiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims, pool='sum',
                 norm_factory=nn.BatchNorm2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        kernel_size = 4 if pool == 'conv' else 1
        self.convs = nn.Sequential(
            norm_factory(in_dims),
            activation_factory(),
            nn.Conv2d(in_dims, out_dims, kernel_size, padding=0, bias=True),
            # nn.Sigmoid()
        )
        self.pool = pool

    def forward(self, img):
        feats = self.convs(img)
        if self.pool == 'avg':
            return F.avg_pool2d(feats, feats.size()[2:]).view(-1, 1)
        elif self.pool == 'sum':
            return torch.sum(feats, [1, 2, 3])[..., None]
        elif self.pool == 'conv':
            print(feats.shape)
            return feats
        else:
            raise ValueError(f'DiscriminatorOutput has no pooling method named "{self.pool}"')


class IQNDiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims,norm_factory=nn.BatchNorm2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        self.to_output = nn.Sequential(
            activation_factory(),
            nn.Linear(in_dims, out_dims),
        )
        feats_dims = in_dims
        self.iqn = IQN(feats_dims)
        self.out_dims = out_dims

    def forward(self, feats, targets=None):
        feats = torch.sum(feats, [2, 3])  # sum pool spatially
        feats_shape = list(feats.shape)
        feats_tau, taus = self.iqn(feats)
        feats_shape[0] = len(feats_tau)
        p_target_tau = self.to_output(feats_tau)
        if targets is not None:
            taus = taus.repeat(1, self.out_dims)
            loss = iqn_loss(p_target_tau, targets, taus)
        p_target_tau = p_target_tau.reshape(self.iqn.num_quantiles, -1, 1)
        p_target = p_target_tau.mean(0)
        if targets is not None:
            return p_target, loss
        return p_target
