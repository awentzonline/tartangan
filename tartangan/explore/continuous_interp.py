import os

import numpy as np
from scipy.stats import truncnorm
import torch
import torchvision

from tartangan.utils.fs import maybe_makedirs
from tartangan.utils.slerp import slerp_grid
from tartangan.trainers.utils import set_device_from_args
from .base import GOutputApp


class ContinuousInterp(GOutputApp):
    """Visualize latent space by blending many output samples per pixel"""
    app_name = "Continuous Interplotion"

    @torch.no_grad()
    def run(self):
        set_device_from_args(self.args)
        self.load_generator()
        if os.path.dirname(self.args.output_prefix):
            maybe_makedirs(os.path.dirname(self.args.output_prefix))
        path = []
        if self.args.tile:
            grid = self.unmirrored_tiled_grid(self.args.num_points, self.args.num_points)
        else:
            grid = self.sample_latent_grid(self.args.num_points, self.args.num_points)
        grid_width, grid_height = grid.shape[:2]
        # grid_imgs = self.g(grid)
        # grid_imgs = grid_imgs.view(
        #     grid.shape[:2] + grid_imgs.shape[-3:]
        # )
        # grid_img_height, grid_img_width = grid_imgs.shape[-2:]
        grid_imgs = {}
        output_width, output_height = self.args.output_size, self.args.output_size
        output_img = torch.zeros(3, output_height, output_width)
        for y in range(output_height):
            print(f'Row {y}')
            grid_y = int(y * grid_height / output_height)
            if not grid_y in grid_imgs:
                row_imgs = self.g(grid[grid_y])
                grid_imgs[grid_y] = row_imgs
            else:
                row_imgs = grid_imgs[grid_y]
            grid_img_height, grid_img_width = row_imgs.shape[-2:]

            img_y = int(y * grid_img_height / output_height)
            for x in range(output_width):
                grid_x = int(x * grid_width / output_width)
                img_x = int(x * grid_img_width / output_width)
                output_img[:, y, x] = row_imgs[grid_x, :, img_y, img_x]
                #output_img[:, y, x] = grid_imgs[grid_x, grid_y, :, img_y, img_x]
        filename = f"{self.args.output_prefix}_combined.png"
        self.save_image(output_img, filename)

    def sample_latent_grid(self, nrows, ncols):
        top_left, top_right, bottom_left, bottom_right = map(
            lambda x: x.cpu(), self.sample_z(4)
        )
        grid = slerp_grid(top_left, top_right, bottom_left, bottom_right, nrows, ncols)
        grid = grid.to(self.args.device)
        return grid.view(nrows, ncols, -1)

    def unmirrored_tiled_grid(self, nrows, ncols):
        nrows = nrows // 3
        ncols = ncols // 3
        a, b, c, d, e, f, g, h, i = map(lambda x: x.cpu(), self.sample_z(9))
        corners = (
            (a, b, c, a),
            (d, e, f, d),
            (g, h, i, g),
            (a, b, c, a),
        )
        all_zs = torch.zeros((nrows - 1) * 3, (ncols - 1) * 3, *a.shape)
        all_offset_row = 0
        for row in range(3):
            all_offset_col = 0
            for col in range(3):
                top_left, top_right = corners[row][col:col + 2]
                bottom_left = corners[row + 1][col]
                bottom_right = corners[row + 1][col + 1]
                grid = slerp_grid(top_left, top_right, bottom_left, bottom_right, nrows, ncols)
                grid = grid.view(nrows, ncols, -1)
                grid = grid[:nrows - 1, :ncols - 1]
                all_zs[all_offset_row:all_offset_row + nrows - 1, all_offset_col:all_offset_col + ncols - 1] = grid
                all_offset_col += ncols - 1
            all_offset_row += nrows - 1
        return all_zs

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('--output-size', default=256, type=int)
        p.add_argument('--num-points', type=int, default=6,
                       help='Number of points to visit')
        p.add_argument('--tile', action='store_true')


if __name__ == '__main__':
    app = ContinuousInterp.create_from_cli()
    app.run()
