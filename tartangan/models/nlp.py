import torch
from torch import nn


class GeneratorNLP(nn.Module):
    def __init__(self, latent_dims, img_size, layer_dims=(32, 64, 128)):
        super().__init__()
        self.img_size = img_size
        self.img_channels = 3
        last_dims = latent_dims
        output_dims = img_size ** 2 * 3
        layers = []
        for dims in layer_dims:
            layers += [
                nn.Linear(last_dims, dims),
                #nn.DropOut(0.3),
                nn.BatchNorm1d(dims),
                nn.ReLU(),
                # ResidualAttentionModule(dims),
                # nn.BatchNorm1d(dims),
                # nn.ReLU(),
            ]
            last_dims = dims
        layers += [
            nn.Linear(last_dims, output_dims),
            nn.Sigmoid(),
        ]
        self.generator = nn.Sequential(*layers)

    def forward(self, z):
        img = self.generator(z)
        batch_size = z.shape[0]
        img = img.view(batch_size, self.img_channels, self.img_size, self.img_size)
        return img


class ResidualAttentionModule(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.x_form = nn.Sequential(
            nn.Linear(in_dims, in_dims)
        )
        self.attention = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.attention(x)
        att_x = att * x
        return x + self.x_form(att_x)


class DiscriminatorNLP(nn.Module):
    def __init__(self, img_size, layer_dims=(128, 64, 32)):
        super().__init__()
        flat_img_dims = img_size ** 2 * 3
        last_dims = flat_img_dims
        layers = []
        for dims in layer_dims:
            layers += [
                nn.Linear(last_dims, dims),
                nn.BatchNorm1d(dims),
                nn.ReLU(),
            ]
            last_dims = dims
        layers += [
            nn.Linear(last_dims, 1),
            nn.Sigmoid(),
        ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, img):
        batch_size = img.shape[0]
        img = img.view(batch_size, -1)
        return self.classifier(img)
