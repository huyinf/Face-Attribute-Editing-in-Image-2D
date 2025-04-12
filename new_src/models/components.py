"""
Neural network components for the ELEGANT model.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class NTimesTanh(nn.Module):
    """Tanh activation scaled by N."""
    
    def __init__(self, N):
        super(NTimesTanh, self).__init__()
        self.N = N
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x) * self.N

class Normalization(nn.Module):
    """Custom normalization layer."""
    
    def __init__(self):
        super(Normalization, self).__init__()
        self.alpha = Parameter(torch.ones(1))
        self.beta = Parameter(torch.zeros(1))

    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=1)
        return x * self.alpha + self.beta

class Encoder(nn.Module):
    """Encoder network for ELEGANT model."""
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 1, bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1, bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 1, bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 2, 1, bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_skip=True):
        """Forward pass through encoder."""
        skip = []
        for i in range(len(self.main)):
            x = self.main[i](x)
            if i < len(self.main) - 1:
                skip.append(x)
        return (x, skip) if return_skip else x

class Decoder(nn.Module):
    """Decoder network for ELEGANT model."""
    
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1, bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, 2, 1, 1, bias=True),
            ),
        ])
        self.activation = NTimesTanh(2)
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, enc1, enc2, skip=None):
        """Forward pass through decoder."""
        x = torch.cat([enc1, enc2], 1)
        for i in range(len(self.main)):
            x = self.main[i](x)
            if skip is not None and i < len(skip):
                x = x + skip[-i-1]
        return self.activation(x)

class Discriminator(nn.Module):
    """Discriminator network for ELEGANT model."""
    
    def __init__(self, n_attributes, img_size):
        super(Discriminator, self).__init__()
        self.n_attributes = n_attributes
        self.img_size = img_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(3+n_attributes, 64, 3, 2, 1, bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(64, 128, 3, 2, 1, bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(128, 256, 3, 2, 1, bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(256, 512, 3, 2, 1, bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.linear = nn.Sequential(
            nn.Linear(512*(self.img_size//16)*(self.img_size//16), 1),
            nn.Sigmoid(),
        )
        
        self.downsample = torch.nn.AvgPool2d(2, stride=2)
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, label):
        """Forward pass through discriminator."""
        while image.shape[-1] != self.img_size or image.shape[-2] != self.img_size:
            image = self.downsample(image)
            
        new_label = label.view(
            (image.shape[0], self.n_attributes, 1, 1)
        ).expand(
            (image.shape[0], self.n_attributes, image.shape[2], image.shape[3])
        )
        
        x = torch.cat([image, new_label], 1)
        output = self.conv(x)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)
        return output 