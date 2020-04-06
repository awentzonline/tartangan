import os

import numpy as np
from scipy.stats import truncnorm
import torch
import torchvision

from tartangan.utils.fs import maybe_makedirs
from tartangan.utils.slerp import slerp
from tartangan.trainers.utils import set_device_from_args
from .base import GOutputApp


class RenderTour(GOutputApp):
    """Renders a circuit of images"""
    app_name = "Render tour"

    def run(self):
        set_device_from_args(self.args)
        self.load_generator()
        path = []
        points = self.sample_z(self.args.num_points)
        for p_a, p_b in zip(points, torch.cat([points[1:], points[0:1]], dim=0)):
            # trim the final 1.0 value from the space:
            for i in np.linspace(0, 1, self.args.seg_frames + 1)[:-1]:
                path.append(slerp(i, p_a, p_b))
        path = torch.tensor(np.stack(path))
        imgs = self.g(path)
        if os.path.dirname(self.args.output_prefix):
            maybe_makedirs(os.path.dirname(self.args.output_prefix))
        for i, img in enumerate(imgs):
            filename = f"{self.args.output_prefix}_{i}.png"
            self.save_image(img, filename)

    @classmethod
    def add_args_to_parser(cls, p):
        p.add_argument('checkpoint_root', help='Path to root of checkpoint files.')
        p.add_argument('output_prefix', help='Prefix for output files.')
        p.add_argument('--num-points', type=int, default=2,
                       help='Number of points to visit')
        p.add_argument('--seg-frames', type=int, default=3,
                       help='Number of points to visit')
        p.add_argument('--trunc-norm', type=float, default=None,
                       help='Sample from truncated normal distribution')


if __name__ == '__main__':
    app = RenderTour.create_from_cli()
    app.run()
