# Model based on: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py
# with changes: https://arxiv.org/abs/2209.15616
#               * Replacing GeLU activations with GeLU
#               * Replacing BatchNorms with GroupNorms
#
# Note, https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md is a fantastic reference for convolution arithmetic.

from collections import OrderedDict
import torch
import torch.nn as nn


class UNet2d(nn.Module):
    """
    U-Net architecture for 2D image segmentation tasks.

    This implementation modifies the traditional U-Net architecture by:
        - Replacing activation functions with GeLU.
        - Replacing Batch Normalization with Group Normalization.
        - Using transposed convolutions for upsampling instead of max pooling.

    Attributes:
    ----------
    in_channels : int
        Number of input channels for the model.
    out_channels : int
        Number of output channels for the model.
    init_features : int
        Number of initial features for the encoder. This value is scaled up by a factor of 2 in each subsequent layer.

    Methods:
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass through the U-Net model.
    _block(in_channels: int, features: int, name: str) -> nn.Sequential
        Creates a U-Net block consisting of two convolutional layers, normalization, and activation.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        """
        Initializes the U-Net model.

        Parameters:
        ----------
        in_channels : int, optional
            Number of input channels (default is 3 for RGB images).
        out_channels : int, optional
            Number of output channels (default is 1 for binary segmentation).
        init_features : int, optional
            Initial number of features (default is 32).
        """
        super(UNet2d, self).__init__()

        features = init_features
        
        # Encoder path
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet2d._block(features * 8, features * 16, name="bottleneck")

        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        # Output layer
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net model.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W), where B is the batch size,
            C is the number of channels, H is the height, and W is the width.

        Returns:
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, H, W) after passing through the U-Net.
        """
        # Encoder
        enc1 = self.encoder1(x)                         # (B, features, H, W)
        enc2 = self.encoder2(self.pool1(enc1))          # (B, features*2, H//2, W//2)
        enc3 = self.encoder3(self.pool2(enc2))          # (B, features*4, H//4, W//4)
        enc4 = self.encoder4(self.pool3(enc3))          # (B, features*8, H//8, W//8)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # (B, features*16, H//16, W//16)

        # Decoder
        dec4 = self.upconv4(bottleneck)                 # (B, features*8, H//8, W//8)
        dec4 = torch.cat((dec4, enc4), dim=1)           # (B, features*16, H//8, W//8)
        dec4 = self.decoder4(dec4)                      # (B, features*8, H//8, W//8)

        dec3 = self.upconv3(dec4)                       # (B, features*4, H//4, W//4)
        dec3 = torch.cat((dec3, enc3), dim=1)           # (B, features*8, H//4, W//4)
        dec3 = self.decoder3(dec3)                      # (B, features*4, H//4, W//4)

        dec2 = self.upconv2(dec3)                       # (B, features*2, H//2, W//2)
        dec2 = torch.cat((dec2, enc2), dim=1)           # (B, features*4, H//2, W//2)
        dec2 = self.decoder2(dec2)                      # (B, features*2, H//2, W//2)

        dec1 = self.upconv1(dec2)                       # (B, features, H, W)
        dec1 = torch.cat((dec1, enc1), dim=1)           # (B, features*2, H, W)
        dec1 = self.decoder1(dec1)                      # (B, features, H, W)

        return self.conv(dec1)                          # (B, out_channels, H, W)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        """
        Creates a U-Net block consisting of two convolutional layers,
        followed by normalization and activation.

        Parameters:
        ----------
        in_channels : int
            Number of input channels for the block.
        features : int
            Number of output channels for the block.
        name : str
            Name prefix used for naming layers within the block.

        Returns:
        -------
        nn.Sequential
            A sequential block containing the defined layers.
        """
        # Each block consists of two convolutional layers with normalization and activation
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(num_groups=1, num_channels=features)),
                    (name + "GeLU1", nn.GELU()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(num_groups=1, num_channels=features)),
                    (name + "GeLU2", nn.GELU()),
                ]
            )
        )
