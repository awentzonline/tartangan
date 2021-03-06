import os

import numpy as np
from PIL import Image
import smart_open
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision import transforms
import tqdm


class ImageBytesDataset(Dataset):
    """Store images as bytes to reduce the memory footprint."""
    def __init__(self, images, transform=None):
        super().__init__()
        self.images = images
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.images.shape[0]

    @classmethod
    def prepare_data_from_path(cls, path, transform=None, trunc=None):
        image_filenames = list_files_recursive(path, IMG_EXTENSIONS)
        if trunc is not None:
            image_filenames = image_filenames[:trunc]
        images = []
        for filename in tqdm.tqdm(image_filenames):
            img = default_loader(filename)
            if transform is not None:
                img = transform(img)
            arr = np.array(img)[None, ...]
            images.append(arr)
        images = np.vstack(images).astype(np.uint8)
        return images

    @classmethod
    def from_path(cls, path, transform=None):
        infile = smart_open.open(path, 'rb')
        images = np.load(infile)
        if isinstance(images, np.lib.npyio.NpzFile):
            images = images['images']
        return cls(images, transform=transform)


def list_files_recursive(root, extensions):
    all_files = []
    for (path, dirs, files) in os.walk(root):
        with_correct_extensions = filter(
            lambda n: os.path.splitext(n)[1] in extensions, files
        )
        all_files.extend([
            os.path.join(path, name) for name in with_correct_extensions
        ])
    return all_files


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Create image data from a folder.')
    p.add_argument('source', help='Root path of images')
    p.add_argument('destination', help='Output location of dataset')
    p.add_argument('--resize', help='Width/height of saved images', default=64, type=int)
    p.add_argument('--trunc', default=None, type=int, help='Take only first N samples')
    p.add_argument('--square', action='store_true', help="Don't preserve aspect ratio")
    args = p.parse_args()

    if args.square:
        resize_shape = (args.resize, args.resize)
        transform = transforms.Compose([
            transforms.Resize(resize_shape, interpolation=Image.LANCZOS),
        ])
    else:
        resize_shape = args.resize
        transform = transforms.Compose([
            transforms.Resize(resize_shape, interpolation=Image.LANCZOS),
            transforms.RandomCrop(resize_shape),
        ])
    print(f'preparing data from "{args.source}"')
    data = ImageBytesDataset.prepare_data_from_path(
        args.source, transform=transform, trunc=args.trunc
    )
    print(f'saving dataset to "{args.destination}"')
    with smart_open.open(args.destination, 'wb') as outfile:
        np.savez_compressed(outfile, images=data)
