import functools

import torch
from torch import nn
import torch.nn.functional as F

from ..layers import Interpolate


class SceneInput(nn.Module):
    def __init__(self, latent_dims, canvas_channels, canvas_size, **_):
        super().__init__()
        self.canvas_shape = (canvas_channels, canvas_size, canvas_size)
        self.empty_canvas = torch.zeros(*self.canvas_shape)

    def forward(self, z):
        canvas = torch.zeros_like(self.empty_canvas)
        canvas = canvas.repeat(z.shape[0], 1, 1, 1)
        return z, canvas


class SceneBlock(nn.Module):
    def __init__(self, z_dims, canvas_channels, patch_size=12,
                 use_alpha=True,
                 norm_factory=nn.BatchNorm1d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2),
                 **kwargs):
        super().__init__()
        self.z_code = nn.Sequential(
            norm_factory(z_dims),
            activation_factory(),
            nn.Linear(z_dims, z_dims),
        )
        self.patch = ScenePatch(z_dims, patch_size, canvas_channels)
        if use_alpha:
            self.alpha = nn.Sequential(
                nn.Linear(z_dims, 1),
                nn.Sigmoid(),
            )
            self.alpha[0].weight.data.zero_()
            self.alpha[0].bias.data.zero_()

        self.use_alpha = use_alpha
        self.refine_canvas = nn.Sequential(
            nn.Conv2d(canvas_channels, canvas_channels, 3, padding=1)
        )

    def forward(self, inputs):
        z, canvas = inputs
        patch_z = self.z_code(z)
        patch = self.patch(patch_z, canvas.size())
        if self.use_alpha:
            alpha = self.alpha(patch_z)[..., None, None]
            canvas = (1 - alpha) * canvas + alpha * patch
        else:
            canvas = canvas + patch
        canvas = self.refine_canvas(canvas)
        z = z - patch_z
        return z, canvas


class ScenePatch(nn.Module):
    def __init__(self, in_dims, patch_size, patch_channels):
        super().__init__()
        self.area = (patch_size ** 2) * patch_channels
        self.patch_size = patch_size
        self.patch_channels = patch_channels
        self.patch = nn.Sequential(
            nn.Linear(in_dims, self.area),
            nn.Tanh(),
        )
        affine_transform_dims = 3 * 2
        self.patch_transform = nn.Sequential(
            nn.Linear(in_dims, affine_transform_dims),
        )
        self.patch_transform[0].weight.data.zero_()
        self.patch_transform[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, b_z, canvas_size):
        patch = self.patch(b_z)
        patch = patch.view(
            -1, self.patch_channels, self.patch_size, self.patch_size
        )
        patch_transform = self.patch_transform(b_z)
        patch_transform = patch_transform.view(-1, 2, 3)
        grid = F.affine_grid(patch_transform, canvas_size, align_corners=True)
        y = F.grid_sample(patch, grid, align_corners=True)
        return y


class SceneUpscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.upscale_canvas = Interpolate(
            scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        z, canvas = inputs
        canvas = self.upscale_canvas(canvas)
        return z, canvas


class SceneOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        z, canvas = inputs
        canvas = torch.tanh(canvas)
        return z, canvas


class SumPool1d(nn.Module):
    def __init__(self, dims=(-1,)):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.sum(x, self.dims)
