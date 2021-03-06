import functools

import torch
from torch import nn
import torch.nn.functional as F

from ..iqn import IQN, iqn_loss
from ..layers import Interpolate


class DiscriminatorInput(nn.Module):
    def __init__(self, in_dims, out_dims,
                 conv_factory=nn.Conv2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        self.convs = nn.Sequential(
            conv_factory(in_dims, out_dims, 1, padding=0, bias=True),
        #    activation_factory(),  # Is this blowing things up?
        )

    def forward(self, img):
        return self.convs(img)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_block=False,
                 norm_factory=nn.BatchNorm2d,
                 conv_factory=nn.Conv2d,
                 avg_pool_factory=nn.AvgPool2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        layers = [
            norm_factory(out_dims),
            activation_factory(),
            conv_factory(in_dims, out_dims, 3, padding=1, bias=True),
            norm_factory(out_dims),
            activation_factory(),
            conv_factory(out_dims, out_dims, 3, padding=1, bias=True),
            avg_pool_factory(2),
        ]
        if first_block:
            layers = layers[2:]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class ResidualDiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_block=False,
                 norm_factory=nn.BatchNorm2d,
                 conv_factory=nn.Conv2d,
                 avg_pool_factory=nn.AvgPool2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2),
                 interpolate=functools.partial(
                    F.interpolate, scale_factor=0.5, mode='bilinear', align_corners=True
                 ),
                 ):
        super().__init__()
        layers = [
            norm_factory(in_dims),
            activation_factory(),
            conv_factory(in_dims, out_dims, 3, padding=1, bias=True),
            norm_factory(out_dims),
            activation_factory(),
            conv_factory(out_dims, out_dims, 3, padding=1, bias=True),
            avg_pool_factory(2),
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
                conv_factory(in_dims, out_dims, 1)
            )
        self.interpolate = interpolate
        # map(nn.init.orthogonal_, self.parameters())

    def _project_input(self, x):
        new_shape = list(x.shape)
        new_shape[1] = self.out_dims
        zs = torch.zeros(*new_shape).to(x.device)
        zs[:, :self.in_dims] = x
        return zs

    def forward(self, x):
        h = self.convs(x)
        x = self.interpolate(x)
        if self.project_input is not None:
            x = self.project_input(x)
        return x + h


class DiscriminatorPoolOnlyOutput(nn.Module):
    def __init__(self, in_dims, out_dims, pool='sum',
                 norm_factory=nn.BatchNorm2d,
                 conv_factory=nn.Conv2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        kernel_size = 4 if pool == 'conv' else 1
        self.convs = nn.Sequential(
            norm_factory(in_dims),
            activation_factory(),
            conv_factory(in_dims, out_dims, kernel_size, padding=0, bias=True),
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


class DiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims,
                 norm_factory=nn.BatchNorm2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2),
                 output_activation_factory=nn.Identity):
        super().__init__()
        self.activation = nn.Sequential(
            norm_factory(in_dims),
            activation_factory(),
        )
        self.to_output = nn.Sequential(
            nn.Linear(in_dims, out_dims),
            output_activation_factory(),
        )

    def forward(self, feats):
        feats = self.activation(feats)
        dims = list(range(2, len(feats.shape)))
        feats = torch.sum(feats, dims)  # sum pool
        y = self.to_output(feats)
        return y


class IQNDiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims, norm_factory=nn.BatchNorm2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        self.activation = nn.Sequential(
            norm_factory(in_dims),
            activation_factory(),
        )
        self.to_output = nn.Sequential(
            nn.Linear(in_dims, out_dims),
        )
        feats_dims = in_dims
        self.iqn = IQN(feats_dims)
        self.out_dims = out_dims

    def forward(self, feats, targets=None):
        feats = self.activation(feats)
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


class MultiModelDiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims, output_model_factories,
                 norm_factory=nn.BatchNorm2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        self.activation = nn.Sequential(
            norm_factory(in_dims),
            activation_factory(),
        )
        self.output_models = nn.ModuleList([
            factory(in_dims)
            for factory in output_model_factories
        ])

    def forward(self, feats):
        feats = self.activation(feats)
        feats = torch.sum(feats, [2, 3])  # sum pool
        ys = [
            model(feats) for model in self.output_models
        ]
        return ys


class LinearOutput(nn.Module):
    def __init__(self, in_dims, out_dims, activation_factory=nn.Identity):
        super().__init__()
        self.xform = nn.Sequential(
            nn.Linear(in_dims, out_dims),
            activation_factory(),
        )

    def forward(self, x):
        return self.xform(x)


class GaussianParametersOutput(nn.Module):
    def __init__(self, in_dims, out_dims,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        self.mu_log_sigma = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            activation_factory(),
            nn.Linear(in_dims, 2 * out_dims)
        )
        self.out_dims = out_dims

    def forward(self, x):
        mu_log_sigma = self.mu_log_sigma(x)
        mu, log_sigma = mu_log_sigma[:, :self.out_dims], mu_log_sigma[:, self.out_dims:]
        return mu, log_sigma
