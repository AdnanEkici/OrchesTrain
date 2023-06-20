from __future__ import annotations

import torch
import torch.nn as nn

from orchestrain.models import base


class DoubleConv(nn.Module):
    def __init__(self, in_chann, out_chann, mid_chann=None):
        super().__init__()

        if mid_chann is None:
            mid_chann = out_chann

        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_channels=in_chann, out_channels=mid_chann, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_chann),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_chann, out_channels=out_chann, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_chann),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.doubleConv(x)


class Down(nn.Module):
    def __init__(self, in_chann, out_chann, mid_chann=None):
        super().__init__()

        if mid_chann is None:
            mid_chann = out_chann

        self.doubleConv = DoubleConv(in_chann=in_chann, mid_chann=mid_chann, out_chann=out_chann)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.doubleConv(x)
        x = self.maxPool(x)
        return x


class Up(nn.Module):
    def __init__(self, in_chann, out_chann):
        super().__init__()

        self.deConv = nn.ConvTranspose2d(
            in_channels=in_chann, out_channels=in_chann // 2, kernel_size=2, stride=2
        )  # Decrease the channel depth to half
        self.doubleConv = DoubleConv(in_chann=in_chann, out_chann=out_chann)  # After concat channel depth will be eq to in_chann

    def forward(self, x1, x2):
        x1 = self.deConv(x1)
        x2 = torch.cat((x1, x2), dim=1)

        return self.doubleConv(x2)


class UNet(base.SemanticSegmentationAdapter):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inc = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        inc = self.inc(x)
        d1 = self.down1(inc)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        x = self.up1(d4, d3)
        x = self.up2(x, d2)
        x = self.up3(x, d1)
        x = self.up4(x, inc)

        return self.conv(x)
