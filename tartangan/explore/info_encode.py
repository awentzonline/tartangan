import glob
import os

from more_itertools import ichunked
from PIL import Image
import torch
from torchvision import transforms
import tqdm

from tartangan.utils.fs import maybe_makedirs
from tartangan.trainers.utils import set_device_from_args
from .base import GOutputApp


class InfoGANEncodeImage(GOutputApp):
    """Encode images using an InfoGAN discriminator"""
    app_name = "InfoGAN image encoder"

    @torch.no_grad()
    def run(self):
        self.setup()

        ids = []
        codes = []
        filename_iter = ichunked(self.gen_filenames(), self.args.batch_size)
        for batch_i, in_filenames in tqdm.tqdm(enumerate(filename_iter)):
            # load up images from a batch of filenames
            batch_imgs = []
            batch_filenames = []
            for in_filename in in_filenames:
                try:
                    target_img = Image.open(in_filename).convert('RGB')
                except IOError:
                    print(f'Error opening {in_filename}')
                    continue
                target_img = self.transform_input_image(target_img)
                batch_imgs.append(target_img)
                batch_filenames.append(os.path.basename(in_filename))

            _, p_code = self.d(torch.stack(batch_imgs).to(self.args.device))

            batch_ids = [os.path.splitext(f)[0] for f in batch_filenames]
            ids += batch_ids
            codes.append(p_code.cpu())

            if self.args.recon:
                recon = self.g(p_code)
                self.save_image(
                    recon, f'{self.args.output_prefix}_{batch_i}.png'
                )
        codes = [c.numpy() for c in codes]
        self.save_codes(ids, codes)

    def save_codes(self, ids, codes):
        import pandas as pd

        codes = [c[i] for c in codes for i in range(len(c))]
        df = pd.DataFrame(dict(id=ids, features=codes))
        print(df.columns)
        print(df.head(100))
        df.to_pickle(f'{self.args.output_prefix}_codes.pkl')

    def gen_filenames(self):
        """
        Yields filenames from the CLI args which are either explict
        filenames or glob expressions.
        """
        for name in self.args.target_images:
            if os.path.isfile(name):
                yield name
            else:
                for filename in glob.iglob(name):
                    yield filename

    def setup(self):
        set_device_from_args(self.args)

        self.load_generator(target=False)
        self.g = self.g.eval()
        self.load_disciminator()
        self.d = self.d.eval()

        img_size = self.g.max_size
        self.transform_input_image = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.LANCZOS),
            transforms.RandomCrop((img_size, img_size)),
            transforms.ToTensor(),
            lambda x: x * 2 - 1
        ])

        if os.path.dirname(self.args.output_prefix):
            maybe_makedirs(os.path.dirname(self.args.output_prefix))

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('target_images', nargs='+',
                       help='Filenames of images to be encoded')
        p.add_argument('--recon', action='store_true',
                       help='Visualize the encoding provided by D')
        p.add_argument('--batch-size', default=32, type=int)


if __name__ == '__main__':
    app = InfoGANEncodeImage.create_from_cli()
    app.run()
