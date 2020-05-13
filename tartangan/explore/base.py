import os

from scipy.stats import truncnorm
import smart_open
import torch
import torchvision

from tartangan.utils.app import App


class GOutputApp(App):
    def sample_z(self, n):
        # TODO: make this part of the model
        if self.args.trunc_norm is not None:
            z = truncnorm.rvs(
                -self.args.trunc_norm, self.args.trunc_norm,
                size=n * self.g.config.latent_dims
            )
            z = z.reshape((n, self.g.config.latent_dims))
            return torch.from_numpy(z).float().to(self.args.device)
        else:
            return torch.randn(n, self.g.config.latent_dims).to(self.args.device)

    def load_generator(self, target=True):
        if os.path.isfile(self.args.checkpoint_root):
            g_filename = self.args.checkpoint_root
        else:
            target_slug = '_target' if target else ''
            g_filename = f'{self.args.checkpoint_root}/g{target_slug}.pt'
        with smart_open.open(g_filename, 'rb') as infile:
            self.g = torch.load(infile, map_location=self.args.device)

    def load_disciminator(self):
        if os.path.isfile(self.args.checkpoint_root):
            filename = self.args.checkpoint_root
        else:
            filename = f'{self.args.checkpoint_root}/d.pt'
        with smart_open.open(filename, 'rb') as infile:
            self.d = torch.load(infile, map_location=self.args.device)

    def save_image(self, img, filename, range=(-1, 1)):
        with smart_open.open(filename, 'wb') as output_file:
            torchvision.utils.save_image(
                img, output_file, normalize=True, range=range, format='png'
            )

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('checkpoint_root', help='Path to root of checkpoint files.')
        p.add_argument('output_prefix', help='Prefix for output files.')
        p.add_argument('--no-cuda', action='store_true')
        p.add_argument('--trunc-norm', type=float, default=None,
                       help='Sample from truncated normal distribution')
