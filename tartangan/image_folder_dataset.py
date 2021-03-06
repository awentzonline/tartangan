import os
import pickle

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from tartangan.utils.fs import maybe_makedirs


class ImageFolderDataset(VisionDataset):
    """Just batches of some images from a folder."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_filenames = list_files_recursive(self.root, IMG_EXTENSIONS)
        self._image_cache = {}

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        if filename not in self._image_cache:
            img = pil_loader(filename)
            xformed = self.transform(img)
            self._image_cache[filename] = xformed
        return self._image_cache[filename]

    def __len__(self):
        return len(self.image_filenames)

    def load_cache(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as infile:
                self._image_cache = pickle.load(infile)

    def save_cache(self, filename):
        if os.path.dirname(filename):
            maybe_makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as outfile:
            pickle.dump(self._image_cache, outfile)


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
