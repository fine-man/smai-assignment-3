import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(
            self, 
            num_channels=8, 
            kernel_size=3, 
            stride=1
        ):
        super(SimpleAutoencoder, self).__init__()

        # Encoder
        in_dim = 28
        k = kernel_size
        s = stride
        pad = (in_dim * (s - 1) + k - s) // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=k, stride=s, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_channels, 2 * num_channels, kernel_size=k, stride=s, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Decoder
        in_channels = 2 * num_channels
        out_channels = num_channels

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # Sigmoid activation for output image reconstruction
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x