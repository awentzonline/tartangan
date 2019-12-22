from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.datasets.utils import list_files
from torchvision import transforms


class JustImagesDataset(VisionDataset):
    """Just batches of some images from a folder."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_filenames = list_files(self.root, IMG_EXTENSIONS, prefix=True)
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
