from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ClassNet(nn.Module):
    def __init__(self, class_nums=10, num_channels=3, num_res_blocks=8, n_feats=64):
        super(ClassNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(num_channels, n_feats, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

        self.in_channels = n_feats

        if num_res_blocks < 3:
            b1 = b2 = b3 = 1
        else:
            b1 = num_res_blocks // 3
            b2 = (num_res_blocks - b1) // 2
            b3 = num_res_blocks - b1 - b2

        self.layer1 = self._make_layer(n_feats, blocks=b1, stride=1)          # 32
        self.layer2 = self._make_layer(n_feats * 2, blocks=b2, stride=2)      # 32 -> 16
        self.layer3 = self._make_layer(n_feats * 4, blocks=b3, stride=2)      # 16 -> 8

        self.tail = nn.Linear(n_feats * 4, class_nums)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        # 第一块下采样
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        # 剩余不变
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.tail(out)

        return out