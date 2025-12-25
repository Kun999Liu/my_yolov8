import torch
import torch.nn as nn


class Conv_maxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Conv5(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, groups=4, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride):  # ch_in, ch_out, stride
        super().__init__()

        self.stride = stride

        branch_features = oup // 2
        band_features = oup // 8
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride == 2:
            # copy input
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=4),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, groups=4, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride == 2) else branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                      groups=4, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=4),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, groups=4, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
        self.branch2_1 = nn.Sequential(
            nn.Conv2d(inp if (self.stride == 2) else band_features, band_features, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(band_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(band_features, band_features, kernel_size=3, stride=self.stride, padding=1, groups=4),
            nn.BatchNorm2d(band_features),

            nn.Conv2d(band_features, band_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(band_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            # x1, x2 = x.chunk(2, dim=1)
            x1, x2, x3, x4 = x.chunk(4, dim=1)
            x1b1, x1b2 = x1.chunk(2, dim=1)
            x2b1, x2b2 = x2.chunk(2, dim=1)
            x3b1, x3b2 = x3.chunk(2, dim=1)
            x4b1, x4b2 = x4.chunk(2, dim=1)
            out1 = torch.cat((x1b1, self.branch2_1(x1b2)), dim=1)
            out2 = torch.cat((x2b1, self.branch2_1(x2b2)), dim=1)
            out3 = torch.cat((x3b1, self.branch2_1(x3b2)), dim=1)
            out4 = torch.cat((x4b1, self.branch2_1(x4b2)), dim=1)
            out = torch.cat((out1, out2, out3, out4), dim=1)
        else:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x1b1, x1b2, x1b3, x1b4 = x1.chunk(4, dim=1)
            x2b1, x2b2, x2b3, x2b4 = x2.chunk(4, dim=1)
            out1 = torch.cat((x1b1, x2b1), dim=1)
            out2 = torch.cat((x1b2, x2b2), dim=1)
            out3 = torch.cat((x1b3, x2b3), dim=1)
            out4 = torch.cat((x1b4, x2b4), dim=1)
            out = torch.cat((out1, out2, out3, out4), dim=1)

        out = self.channel_shuffle(out, 2)

        return out

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

        return out
