import torch
import torch.nn.functional as F


# Found these hinge loss functions in this BigGAN repo:
# https://github.com/ajbrock/BigGAN-PyTorch
def discriminator_hinge_loss(real, fake):
    loss_real = torch.mean(F.relu(1. - real))
    loss_fake = torch.mean(F.relu(1. + fake))
    return loss_real, loss_fake


def generator_hinge_loss(fake):
    return -torch.mean(fake)
