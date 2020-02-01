import torch
from torch import nn
import torch.nn.functional as F

from .iqn import IQN, iqn_loss


class GeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True, norm_factory=nn.BatchNorm2d):
        super().__init__()
        layers = [
            # nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            # ),
            nn.LeakyReLU(0.2),
            norm_factory(out_dims),
            # nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True),
            # ),
            nn.LeakyReLU(0.2),
            norm_factory(out_dims),
        ]
        if upsample:
            layers.insert(0, Interpolate(scale_factor=2, mode='bilinear', align_corners=True))
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ResidualGeneratorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, upsample=True, norm_factory=nn.BatchNorm2d):
        super().__init__()
        layers = [
            #nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            #),
            nn.LeakyReLU(0.2),
            norm_factory(out_dims),
            #nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True),
            #),
            nn.LeakyReLU(0.2),
            norm_factory(out_dims),
        ]
        self.upsample = upsample
        self.project_input = None
        if in_dims != out_dims:
            self.project_input = nn.Sequential(
                #nn.utils.spectral_norm(
                    nn.Conv2d(in_dims, out_dims, 1)
                #),
            )
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        h = self.convs(x)
        if self.project_input is not None:
            x = self.project_input(x)
        return F.relu(x + h)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_conv=None, norm_factory=nn.BatchNorm2d):
        super().__init__()
        layers = [
            #nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            #),
            nn.LeakyReLU(0.2, inplace=True),
            norm_factory(out_dims),
            #nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True),
            #),
            nn.LeakyReLU(0.2, inplace=True),
            norm_factory(out_dims),
            Interpolate(scale_factor=0.5, mode='bilinear', align_corners=True),
        ]
        self.convs = nn.Sequential(*layers)
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class ResidualDiscriminatorBlock(nn.Module):
    def __init__(self, in_dims, out_dims, first_conv=None, norm_factory=nn.BatchNorm2d):
        super().__init__()
        layers = [
            #nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 3, padding=1, bias=True),
            #),
            nn.LeakyReLU(0.2, inplace=True),
            norm_factory(out_dims),
            #nn.utils.spectral_norm(
                nn.Conv2d(out_dims, out_dims, 3, padding=1, bias=True),
            #),
            norm_factory(out_dims),
            #nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=0.5, mode='bilinear', align_corners=True),
        ]
        self.no_skip = bool(first_conv)
        self.convs = nn.Sequential(*layers)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.project_input = None
        if in_dims != out_dims:
            self.project_input = self._project_input
        map(nn.init.orthogonal_, self.parameters())

    def _project_input(self, x):
        new_shape = list(x.shape)
        new_shape[1] = self.out_dims
        zs = torch.zeros(*new_shape).to(x.device)
        zs[:, :self.in_dims] = x
        return zs

    def forward(self, x):
        h = self.convs(x)
        # if self.no_skip:
        #     return h
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        if self.project_input is not None:
            x = self.project_input(x)
        return F.leaky_relu(x + h, 0.2)


class GeneratorInputMLP(nn.Module):
    def __init__(self, latent_dims, size=4, norm_factory=nn.BatchNorm1d):
        super().__init__()
        base_img_dims = size ** 2 * latent_dims
        self.base_img = nn.Sequential(
            #nn.utils.spectral_norm(
                nn.Linear(latent_dims, base_img_dims),
            #),
            nn.ReLU(),
            norm_factory(base_img_dims),
        )
        self.latent_dims = latent_dims
        self.size = size

    def forward(self, z):
        img = self.base_img(z)
        return img.view(-1, self.latent_dims, self.size, self.size)


class TiledZGeneratorInput(nn.Module):
    def __init__(
        self, latent_dims, size=4, norm_factory=nn.BatchNorm2d,
    ):
        super().__init__()
        self.size = size

    def forward(self, z):
        components = z[..., None, None].repeat(1, 1, self.size, self.size)
        return components


class WeightedComponents(nn.Module):
    def __init__(self, latent_dims, size=4, num_components=None, norm_factory=nn.BatchNorm2d):
        super().__init__()
        num_components = latent_dims if num_components is None else num_components
        self.components = nn.Parameter(
            torch.randn(1, num_components, size, size, requires_grad=True) * 0.1
            , requires_grad=True
        )
        self.img_weights = nn.Sequential(
            # nn.utils.spectral_norm(
                 nn.Linear(latent_dims, num_components),
            # ),
            # nn.LeakyReLU(0.2),
            #nn.utils.spectral_norm(
            #    nn.Linear(num_components, num_components),
            #),
            #nn.Sigmoid(),
        )
        self.output_norm = norm_factory(latent_dims)

    def forward(self, z):
        weights = self.img_weights(z)[..., None, None]
        weighted = self.components * weights
        return self.output_norm(F.leaky_relu(weighted, 0.2))


class GeneratorOutput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            #nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True),
            #),
            nn.Tanh()  # nn.Sigmoid()
        )
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, x):
        return self.convs(x)


class SpineBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        pass


class DiscriminatorInput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            #nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True),
            #),
            nn.LeakyReLU(0.2),
        )
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, img):
        return self.convs(img)


class DiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            # nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True),
            # ),
            # nn.Tanh()
            nn.Sigmoid()
        )
        map(nn.init.orthogonal_, self.parameters())

    def forward(self, img):
        feats = self.convs(img)
        return F.avg_pool2d(feats, feats.size()[2:]).view(-1, 1)


class IQNDiscriminatorOutput(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.convs = nn.Sequential(
            #nn.utils.spectral_norm(
                nn.Conv2d(in_dims, out_dims, 1, padding=0, bias=True)
            #),
            #nn.Tanh()#Sigmoid()
        )
        map(nn.init.orthogonal_, self.parameters())
        # avoid ortho init for IQN
        feats_dims = 2 * 2 * in_dims
        self.iqn = IQN(feats_dims)
        self.out_dims = out_dims

    def forward(self, feats, targets=None):
        feats_shape = list(feats.shape)
        feats = feats.view(feats_shape[0], -1)
        feats_tau, taus = self.iqn(feats)
        feats_shape[0] = len(feats_tau)
        feats_tau = feats_tau.view(*feats_shape)
        feats = self.convs(feats_tau)
        p_target_tau = F.avg_pool2d(feats, feats.size()[2:]).view(-1, self.out_dims)
        if targets is not None:
            taus = taus.repeat(1, self.out_dims)
            loss = iqn_loss(p_target_tau, targets, taus)
        p_target_tau = p_target_tau.reshape(self.iqn.num_quantiles, -1, 1)
        p_target = p_target_tau.mean(0)
        if targets is not None:
            return p_target, loss
        return p_target


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt((x ** 2).mean(1, keepdim=True) + self.eps)
