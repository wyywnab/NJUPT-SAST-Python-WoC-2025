from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return residual + out * 0.1

class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels * (scale_factor ** 2), kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class SRNet(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, num_res_blocks=16, n_feats=64):
        super(SRNet, self).__init__()

        self.head = nn.Conv2d(num_channels, n_feats, kernel_size=3, padding=1, bias=True)

        self.body = nn.Sequential(
            *[ResidualBlock(n_feats) for _ in range(num_res_blocks)]
        )
        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)

        self.upsample = UpsampleBlock(n_feats, scale_factor)

        self.tail = nn.Conv2d(n_feats, num_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res = self.conv(res)
        res += x

        out = self.upsample(res)
        out = self.tail(out)

        return out