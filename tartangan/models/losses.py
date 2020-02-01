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


# grads_squared_norm = tf.pow(tf.gradients(tf.reduce_sum(logits, axis=0), data)[0], 2, name='grads_squared_norm')
# grads_squared_norm = tf.reduce_sum(tf.reshape(grads_squared_norm, [data.get_shape()[0], -1]), axis=1)
# return gamma_gp * tf.reduce_mean(grads_squared_norm, name='gp_loss')

def gradient_penalty(preds, data):
    """
    https://arxiv.org/pdf/1801.04406.pdf
    https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/4
    """
    outputs = torch.ones_like(preds).to(preds.device)
    data.requires_grad_()
    gradients = torch.autograd.grad(
        outputs=preds.sum(1, keepdim=True), inputs=data,
        grad_outputs=outputs,
        retain_graph=True, create_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm_sq = gradients.norm(2, dim=1) ** 2
    return gradient_norm_sq.mean()


# def gradient_penalty(preds, data):
#     data.requires_grad_()
#     grad_real = torch.autograd.grad(outputs=preds.sum(), inputs=data, create_graph=True)[0]
#     grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
#     # grad_penalty_real = 10 / 2 * grad_penalty_real
#     return grad_penalty_real
