import torch
from torch import nn
import torch.nn.functional as F

from .iqn import IQN, iqn_loss
from .layers import Interpolate, PixelNorm


class GeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True, first_block=False,
                 norm_factory=nn.BatchNorm2d, activation_factory=nn.LeakyReLU):
        super().__init__()
        layers = [
            norm_factory(out_dims),
            activation_factory(),
            nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            norm_factory(out_dims),
            activation_factory(),
            nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True),
        ]
        if first_block:
            layers = layers[2:]
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='bilinear', align_corners=True))
        self.convs = nn.Sequential(*layers)
        # map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ResidualGeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True, first_block=False,
                 norm_factory=nn.BatchNorm2d, activation_factory=nn.LeakyReLU):
        super().__init__()

        layers = [
            norm_factory(in_dims),
            activation_factory(),
            nn.Conv2d(in_dims, out_dims, 3, padding=1),
            norm_factory(out_dims),
            activation_factory(),
            nn.Conv2d(out_dims, out_dims, 3, padding=1),
        ]
        if first_block:
            layers = layers[2:]
        self.upsample = upsample
        self.project_input = None
        if in_dims != out_dims:
            self.project_input = nn.Sequential(
                nn.Conv2d(in_dims, out_dims, 1)
            )
        self.convs = nn.Sequential(*layers)
        # map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        h = self.convs(x)
        if self.project_input is not None:
            x = self.project_input(x)
        return x + h


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_block=False, norm_factory=nn.BatchNorm2d):
        super().__init__()
        layers = [
            norm_factory(out_dims),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            norm_factory(out_dims),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True),
            Interpolate(scale_factor=0.5, mode='bilinear', align_corners=True),
        ]
        if first_block:
            layers = layers[2:]
        self.convs = nn.Sequential(*layers)
        # map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ResidualDiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_block=False, norm_factory=nn.BatchNorm2d):
        super().__init__()
        layers = [
            norm_factory(in_dims),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            norm_factory(out_dims),
            nn.LeakyReLU(0.2),
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


class GeneratorInputMLP(nn.Module):
    def __init__(self, latent_dims, output_dims, size=4, norm_factory=nn.BatchNorm1d):
        super().__init__()
        base_img_dims = size ** 2 * output_dims
        self.base_img = nn.Sequential(
            nn.Linear(latent_dims, base_img_dims),
            nn.LeakyReLU(0.2),
        )
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.size = size

    def forward(self, z):
        img = self.base_img(z)
        return img.view(-1, self.output_dims, self.size, self.size)


class TiledZGeneratorInput(nn.Module):
    def __init__(
        self, latent_dims, output_dims, size=4, norm_factory=nn.BatchNorm2d,
    ):
        super().__init__()
        self.size = size
        assert latent_dims == output_dims

    def forward(self, z):
        components = z[..., None, None].repeat(1, 1, self.size, self.size)
        return components


class GeneratorOutput(nn.Module):
    def __init__(self, in_dims, out_dims, norm_factory=nn.Identity,#nn.BatchNorm2d,
                 activation_factory=nn.LeakyReLU):
        super().__init__()
        self.convs = nn.Sequential(
            norm_factory(in_dims),
            activation_factory(),
            nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True),
            nn.Tanh()
        )
        # map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class DiscriminatorInput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True),
            # nn.LeakyReLU(0.2),
        )
        # map(nn.init.orthogonal_, self.parameters())

    def forward(self, img):
        return self.convs(img)


class DiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims, norm_factory=nn.BatchNorm2d, pool='sum'):
        super().__init__()
        kernel_size = 4 if pool == 'conv' else 1
        self.convs = nn.Sequential(
            norm_factory(in_dims),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dims, out_dims, kernel_size, padding=0, bias=True),
        #    nn.Sigmoid()
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
    def __init__(self, in_dims, out_dims, norm_factory=nn.BatchNorm2d):
        super().__init__()
        self.to_output = nn.Sequential(
            nn.LeakyReLU(0.2),
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
