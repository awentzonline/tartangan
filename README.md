tartangan
=========
Use a GAN to model tartan patterns. Big ol' work in progress.


Preparing a Dataset
-----------------
Here's [an archive of tartan images](https://github.com/awentzonline/tartangan/releases/download/0.0.0files/tartan_images.zip) to get you started.

There are two ways of providing your dataset:

ImageFolderDataset: This loader will lazily resize the images the first time
they're used in training. This is easy to use but is slower during the first
epoch due to the loading and resizing being done at run-time. Simply pass the
root path of your images to the trainer as the `dataset` argument.

ImageBytesDataset: Loads an archive of images which have already been resized
so it doesn't have the first epoch slowness. To prepare your images for use with
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

Test metrics
------------
I've adapted code from the excellent [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) to calculate the Inception Score and Frechet Inception Distance, popular measures of how closely the generator output matches the real data.

First precalculate the inception moments of the dataset with:

`python3 -m tartangan.calculate_inception_moments $DATASET $IM_FILENAME`

Then run the trainer with the `--inception-moments=$IM_FILENAME` argument. Calculating these takes a bit of time. You can choose how frequently to run the tests with `--test-freq` and the number of samples drawn from the generator with `--n-inception-imgs`.

The scores will be printed during training and you can choose to have it output final scores to a JSON-encoded file with `--metrics-path`. Currently, this file is formatted for use with scalar metrics in a [Kubeflow Pipeline](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/).

Resume from checkpoint
----------------------
To resume training, specify the CLI arguments `--run-id` and `--resume-training-step`
`run_id` is the path segment which looks like a datetime with a random suffix and
`resume-training-step` should be a number that appears in the checkpoints directory.
