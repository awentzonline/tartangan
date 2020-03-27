''' Calculate Inception Moments
 Adapted from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/calculate_inception_moments.py
 under the MIT license.

 This script iterates over the dataset and calculates the moments of the
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.

 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import numpy as np
import smart_open
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchvision import transforms

from .image_bytes_dataset import ImageBytesDataset
from .trainers.tqdm_newlines import tqdm_class, tqdm_kwargs
from . import inception_utils


def calculate_inception_moments(loader, use_newlines=False, log_iters=10, quiet_logs=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device "{device}"')
    print('Loading inception net...')
    net = inception_utils.load_inception_net(parallel=False)
    net = net.to(device)
    pool, logits = [], []
    print('Evaluating dataset activations...')
    tqdm = tqdm_class(use_newlines)
    progress_iter = tqdm(loader, **tqdm_kwargs(quiet_logs, log_iters))
    for i, x in enumerate(progress_iter):
        x = x.to(device)
        with torch.no_grad():
            pool_val, logits_val = net(x)
            pool += [np.asarray(pool_val.cpu())]
            logits += [np.asarray(F.softmax(logits_val, 1).cpu())]

    pool, logits = [np.concatenate(item, 0) for item in [pool, logits]]

    print('Calculating inception metrics...')
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    print('Training data from dataset has IS of %5.5f +/- %5.5f' % (IS_mean, IS_std))

    # Prepare mu and sigma, save to disk. Remove "hdf5" by default
    # (the FID code also knows to strip "hdf5")
    print('Calculating means and covariances...')
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    return mu, sigma


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Create image data from a folder.')
    p.add_argument('source', help='Root path of dataset')
    p.add_argument('destination', help='Output location')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--log-iters', type=int, default=10)
    p.add_argument('--quiet-logs', action='store_true',
                   help='Less verbose logs')
    p.add_argument('--log-newlines', action='store_true',
                   help='Use newlines instead of carriage returns in progress bar.')
    args = p.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    print(f'Loading dataset from {args.source}')
    dataset = ImageBytesDataset.from_path(
        args.source, transform=transform
    )
    loader = data_utils.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    mu, sigma = calculate_inception_moments(
        loader, use_newlines=args.log_newlines, quiet_logs=args.quiet_logs,
        log_iters=args.log_iters
    )
    print(f'Saving calculated means and covariances to "{args.destination}"...')
    with smart_open.open(args.destination, 'wb') as outfile:
        np.savez(outfile, mu=mu, sigma=sigma)
