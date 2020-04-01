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
        self.refine_canvas = nn.Sequential(
            nn.Conv2d(canvas_channels, canvas_channels, 3, padding=1)
        )

    def forward(self, inputs):
        z, canvas = inputs
        patch_z = self.z_code(z)
        patch, mask = self.patch(patch_z, canvas.size())
        canvas = (1 - mask) * canvas + patch
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
        self.alpha = nn.Sequential(
            nn.Linear(in_dims, self.area),
            nn.Sigmoid(),
        )
        self.alpha[0].weight.data.zero_()
        self.alpha[0].bias.data.zero_()

        affine_transform_dims = 6
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
        alpha = self.alpha(b_z)
        alpha = alpha.view(
            -1, self.patch_channels, self.patch_size, self.patch_size
        )
        patch = patch * alpha
        patch_transform = self.patch_transform(b_z)
        patch_transform = patch_transform.view(-1, 2, 3)
        grid = F.affine_grid(patch_transform, canvas_size, align_corners=True)
        y = F.grid_sample(patch, grid, align_corners=True)
        mask = F.grid_sample(alpha, grid, align_corners=True)
        return y, mask


class SceneStructureBlock(nn.Module):
    """Generates the structure of a scene.

    Outputs 2d maps of opacity and orientation per component
    """
    def __init__(self, in_dims, num_patches, patch_size=3, scene_size=16,
                 output_orientations=False, refine_patches=False,
                 patch_noise=True,
                 norm_factory=nn.BatchNorm1d,
                 activation_factory=functools.partial(nn.LeakyReLU, 0.2),
                 **kwargs):
        super().__init__()
        self.patch_area = patch_size ** 2
        self.masks = nn.Sequential(
            nn.Linear(in_dims, num_patches * self.patch_area),
            nn.Sigmoid(),
        )
        self.masks[0].weight.data.zero_()
        self.masks[0].bias.data.zero_()

        affine_transform_dims = 2 * 3
        self.patch_transforms = nn.Sequential(
            nn.Linear(in_dims, affine_transform_dims * num_patches),
        )
        self.patch_transforms[0].weight.data.zero_()
        initial_scale = 2
        self.patch_transforms[0].bias.data.copy_(
            torch.tensor(
                [initial_scale, 0, 0, 0, initial_scale, 0],
                dtype=torch.float
            ).repeat(num_patches)
        )

        self.num_patches = num_patches
        self.output_orientations = output_orientations
        self.scene_size = scene_size
        self.patch_size = patch_size
        self.patch_noise = patch_noise
        if patch_noise:
            self.noise_proto = nn.Parameter(
                torch.zeros(self.patch_size, self.patch_size),
                requires_grad=False
            )
        self.refine_patches = refine_patches
        if not refine_patches:
            self.full_masks = nn.Parameter(
                torch.ones(self.num_patches, self.patch_size, self.patch_size),
                requires_grad=False
            )

    def forward(self, z):
        if self.refine_patches:
            masks = (1. - self.masks(z))  # starts at zero, which should be opaque
            masks = masks.view(-1, self.num_patches, self.patch_size, self.patch_size)
        else:
            masks = torch.ones_like(self.full_masks)[None, ...].repeat(z.shape[0], 1, 1, 1)
        transforms = self.patch_transforms(z)
        transforms = transforms.view(-1, self.num_patches, 2, 3)
        masks = masks.permute(1, 0, 2, 3)  # PBHW
        transforms = transforms.permute(1, 0, 2, 3)
        patches = []
        if self.patch_noise:
            noise = torch.randn_like(self.noise_proto)
        for i in range(self.num_patches):
            mask = masks[i][:, None, ...]
            if self.patch_noise:
                mask = mask * noise
            transform = transforms[i]
            grid = F.affine_grid(transform, (z.shape[0], 1, self.scene_size, self.scene_size), align_corners=False)
            transformed_mask = F.grid_sample(mask, grid, align_corners=False)
            transformed_mask = transformed_mask.squeeze(1)
            patches.append(transformed_mask)  # [BHW, BHW, ...]

        patches = torch.stack(patches, dim=0)  # PBHW
        patches = patches.permute(1, 0, 2, 3)  # BPHW
        return patches

    @property
    def output_channels(self):
        return self.num_patches# + self.output_orientations * 4


class SceneUpscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.upscale_canvas = Interpolate(
            scale_factor=2, mode='nearest', align_corners=False)

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
