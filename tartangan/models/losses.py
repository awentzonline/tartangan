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


def gradient_penalty(preds, data):
    """
    https://arxiv.org/pdf/1801.04406.pdf
    https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
    """
    batch_size = data.size(0)
    grad_dout = torch.autograd.grad(
        outputs=preds.sum(), inputs=data,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == data.size())
    reg = grad_dout2.view(batch_size, -1).sum(1).mean()
    return reg
