# raw unet architecture
from torchinfo import summary
import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    '''
    returns a masked 1 channel tensor of the same shape as the input tensor
    '''

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_bottom = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)

        self.upsample4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dconv_up4 = double_conv(1024, 512)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_bottom(x)
        x = self.upsample4(x)

        crop_conv4 = conv4[:, :, :x.size(2), :x.size(3)]
        x = torch.cat([x, crop_conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample3(x)

        crop_conv3 = conv3[:, :, :x.size(2), :x.size(3)]
        x = torch.cat([x, crop_conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample2(x)

        crop_conv2 = conv2[:, :, :x.size(2), :x.size(3)]
        x = torch.cat([x, crop_conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)

        crop_conv1 = conv1[:, :, :x.size(2), :x.size(3)]
        x = torch.cat([x, crop_conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)

        return out


# test run
model = UNet().to('cpu')
summary(model, input_size=(32, 1, 256, 256))
