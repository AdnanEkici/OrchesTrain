from __future__ import annotations

import torch
import torch.nn as nn

from orchestrain.models import base


class InitialLayer(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.conv(x)
        skip = torch.cat([x] * out.shape[1], 1)
        return self.relu(torch.add(skip, out))


class ConvBlock(nn.Module):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.PReLU(n_channels)
        self.conv = nn.Conv3d(n_channels, n_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(n_channels)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv, dropout=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(out_channels),
        )
        self.dropuot = nn.Dropout3d() if dropout else nn.Identity()
        self.n_convs = nn.Sequential(*[ConvBlock(out_channels) for _ in range(n_conv)])
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        down = self.down(x)
        out = self.dropuot(down)
        out = self.n_convs(out)
        return self.relu(torch.add(out, down))


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv, dropout=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels // 2),
            nn.PReLU(out_channels // 2),
        )
        self.drop_out1 = nn.Dropout3d() if dropout else nn.Identity()
        self.drop_out2 = nn.Dropout3d()
        self.n_convs = nn.Sequential(*[ConvBlock(out_channels) for _ in range(n_conv)])
        self.relu = nn.PReLU(out_channels)

    def forward(self, x, fine_grained):
        upsample = self.up(self.drop_out1(x))
        fine_grained = self.drop_out2(fine_grained)

        fined_upsample = torch.cat((upsample, fine_grained), dim=1)
        out = self.n_convs(fined_upsample)
        return self.relu(torch.add(out, fined_upsample))


class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.last_conv(x)


class VNet(base.SemanticSegmentationAdapter):
    def __init__(self, in_channels, out_channels, dims=[16, 32, 64, 128, 256], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = InitialLayer(in_channels, dims[0])
        self.down1 = DownConv(dims[0], dims[1], 2)
        self.down2 = DownConv(dims[1], dims[2], 3)
        self.down3 = DownConv(dims[2], dims[3], 3, dropout=True)
        self.down4 = DownConv(dims[3], dims[4], 3, dropout=True)

        self.up1 = UpConv(dims[4], dims[4], 3)
        self.up2 = UpConv(dims[4], dims[3], 3)
        self.up3 = UpConv(dims[3], dims[2], 2)
        self.up4 = UpConv(dims[2], dims[1], 1)

        self.last = OutputLayer(dims[1], out_channels)

    def forward(self, x):
        fine_grained16 = self.input_layer(x)
        fine_grained32 = self.down1(fine_grained16)
        fine_grained64 = self.down2(fine_grained32)
        fine_grained128 = self.down3(fine_grained64)

        bottom256 = self.down4(fine_grained128)

        fed256 = self.up1(x=bottom256, fine_grained=fine_grained128)
        fed128 = self.up2(fed256, fine_grained=fine_grained64)
        fed64 = self.up3(x=fed128, fine_grained=fine_grained32)
        fed32 = self.up4(x=fed64, fine_grained=fine_grained16)
        out = self.last(fed32)
        return out
