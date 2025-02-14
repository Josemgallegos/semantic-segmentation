import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
        )

        self.middle = DoubleConv(128, 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(64, 64),
        )

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)  # Capa de salida

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return torch.sigmoid(self.final(x3))  # Activación sigmoide para segmentación binaria