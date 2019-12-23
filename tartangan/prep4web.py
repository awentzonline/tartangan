import argparse

import torch
from torch import nn


def package_for_web(model, filename, batch_size=1):
    remove_all_spectral_norm(model)
    dummy_input = torch.randn(batch_size, model.config.latent_dims)
    torch.onnx.export(
        model, dummy_input, filename, verbose=True,
    )


def remove_all_spectral_norm(item):
    """https://github.com/pytorch/pytorch/issues/27723"""
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass

        for child in item.children():
            remove_all_spectral_norm(child)

    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)

    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--output', default='ttgan.onnx')
    p.add_argument('--batch-size', default=1, type=int)
    args = p.parse_args()

    generator = torch.load(args.checkpoint, map_location='cpu')
    package_for_web(generator, args.output)


if __name__ == '__main__':
    main()
