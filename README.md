tartangan
=========
Use a GAN to model tartan patterns. Big ol' work in progress.


Preparing Dataset
-----------------
Here's [an archive of tartan images](https://github.com/awentzonline/tartangan/releases/download/0.0.0files/tartan_images.zip)

There are two ways of providing your dataset:

ImageFolderDataset: This loader will lazily resize the images the first time
they're used in training. This is easy to use but is slower during the first
epoch due to the loading and resizing being done at run-time. Simply pass the
root path of your images to the trainer as the `dataset` argument.

ImageBytesDataset: Loads an archive of images which have already been resized
so it doesn't have the 1st epoch slowness. To prepare your images for use with
this loader run:

`python -m tartangan.image_bytes_dataset --resize=$SIZE $PATH_TO_IMAGES $OUTPUT_FILENAME`

Then pass `$OUTPUT_FILENAME` to the trainer as the `dataset` argument.

Training
--------
 * Clone and install this repo
 * Optionally pre-resize your dataset as described in the `Preparing Dataset` section
 * `python -m tartangan.trainers.cnn $DATASET` to train a SA-GAN alike model
 * `python -m tartangan.trainers.iqn $DATASET` to train a SA-GAN-IQN model
 * There are many CLI options for the trainer. Run with `--help` for more information.
