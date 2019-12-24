import torch
from torch import nn
import torch.nn.functional as F

from .iqn import IQN, iqn_loss


class GeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
        ]
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='bilinear', align_corners=True))
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class GeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
        ]
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='bilinear', align_corners=True))
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ResidualGeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
        ]
        self.upsample = upsample
        self.project_input = None
        if in_dims != out_dims:
            self.project_input = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_dims, out_dims, 1)),
            )
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        h = self.convs(x)
        if self.project_input is not None:
            x = self.project_input(x)
        return x + h


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_conv=None):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=0.5, mode='bilinear', align_corners=True),
        ]
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ResidualDiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_conv=None):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True)),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=0.5, mode='bilinear', align_corners=True),
        ]
        self.no_skip = bool(first_conv)
        self.convs = nn.Sequential(*layers)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.project_input = None
        if in_dims != out_dims:
            self.project_input = self._project_input
        map(nn.init.orthogonal_, self.parameters())

    def _project_input(self, x):
        new_shape = list(x.shape)
        new_shape[1] = self.out_dims
        zs = torch.zeros(*new_shape).to(x.device)
        zs[:, :self.in_dims] = x
        return zs

    def forward(self, x):
        h = self.convs(x)
        # if self.no_skip:
        #     return h
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        if self.project_input is not None:
            x = self.project_input(x)
        return x + h


class GeneratorOutput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class DiscriminatorInput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True)),
                nn.ReLU(),
        )
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, img):
        return self.convs(img)


class DiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True)),
            #nn.Tanh()#Sigmoid()
        )
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, img):
        feats = self.convs(img)
        return F.avg_pool2d(feats, feats.size()[2:]).view(-1, 1)


class IQNDiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True)),
            #nn.Tanh()#Sigmoid()
        )
        map(nn.init.orthogonal_, self.parameters())
        # avoid ortho init for IQN
        feats_dims = 2 * 2
        self.iqn = IQN(feats_dims)

    def forward(self, img, targets=None):
        feats = self.convs(img)
        feats_shape = list(feats.shape)
        feats = feats.view(feats_shape[0], -1)
        feats_tau, taus = self.iqn(feats)
        feats_shape[0] = len(feats_tau)
        feats_tau = feats_tau.view(*feats_shape)
        p_target_tau = F.avg_pool2d(feats_tau, feats_tau.size()[2:]).view(-1, 1)
        if targets is not None:
            loss = iqn_loss(p_target_tau, targets, taus)
        p_target_tau = p_target_tau.reshape(self.iqn.num_quantiles, -1, 1)
        p_target = p_target_tau.mean(0)
        if targets is not None:
            return p_target, loss
        return p_target


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)
