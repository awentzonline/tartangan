import functools

from torch import nn
import torch.nn.functional as F

from ..layers import Interpolate


class GeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True, first_block=False,
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
        ]
        if first_block:
            layers = layers[2:]
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='nearest'))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class ResidualGeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True, first_block=False,
                 norm_factory=nn.BatchNorm2d, conv_factory=nn.Conv2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()

        layers = [
            norm_factory(in_dims),
            activation_factory(),
            conv_factory(in_dims, out_dims, 3, padding=1),
            norm_factory(out_dims),
            activation_factory(),
            conv_factory(out_dims, out_dims, 3, padding=1),
        ]
        if first_block:
            layers = layers[2:]
        self.upsample = upsample
        self.project_input = None
        if in_dims != out_dims:
            self.project_input = nn.Sequential(
                conv_factory(in_dims, out_dims, 1)
            )
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        h = self.convs(x)
        if self.project_input is not None:
            x = self.project_input(x)
        return x + h


class GeneratorInputMLP(nn.Module):
    def __init__(self, latent_dims, output_dims, size=4, norm_factory=nn.BatchNorm1d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        base_img_dims = size ** 2 * output_dims
        self.base_img = nn.Sequential(
            nn.Linear(latent_dims, base_img_dims),
            activation_factory(),
        )
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.size = size

    def forward(self, z):
        img = self.base_img(z)
        return img.view(-1, self.output_dims, self.size, self.size)


class GeneratorInputMLP1d(nn.Module):
    def __init__(self, latent_dims, output_dims, size=4, norm_factory=nn.BatchNorm1d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2)):
        super().__init__()
        base_img_dims = size * output_dims
        self.base = nn.Sequential(
            nn.Linear(latent_dims, base_img_dims),
            activation_factory(),
        )
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.size = size

    def forward(self, z):
        img = self.base(z)
        return img.view(-1, self.output_dims, self.size)


class TiledZGeneratorInput(nn.Module):
    def __init__(
        self, latent_dims, output_dims, size=4, norm_factory=nn.BatchNorm2d,
        **_
    ):
        super().__init__()
        self.size = size
        assert latent_dims == output_dims

    def forward(self, z):
        components = z[..., None, None].repeat(1, 1, self.size, self.size)
        return components


class GeneratorOutput(nn.Module):
    def __init__(self, in_dims, out_dims, norm_factory=nn.BatchNorm2d,
                 conv_factory=nn.Conv2d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2),
                 output_activation_factory=nn.Tanh):
        super().__init__()
        self.convs = nn.Sequential(
            norm_factory(in_dims),
            activation_factory(),
            conv_factory(in_dims, out_dims, 1, padding=0, bias=True),
            output_activation_factory(),
        )

    def forward(self, x):
        return self.convs(x)
