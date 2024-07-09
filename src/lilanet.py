import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F

# from padding import conv2d_get_padding


class EvenBlock(nn.Module):
    def __init__(self, in_channels, n, modified=False):
        super(EvenBlock, self).__init__()
        self.pad_input = None
        self.branch1 = BasicConv2d(
            in_channels, n, kernel_size=(7, 3), padding=(2, 0))
        self.branch2 = BasicConv2d(in_channels, n, kernel_size=3)
        self.branch3 = BasicConv2d(
            in_channels, n, kernel_size=(3, 7), padding=(0, 2))
        # self.branch4 = BasicConv2d(in_channels, n, kernel_size=3, dilation=2)
        self.conv = BasicConv2d(n * 3, n, kernel_size=1, padding=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        # self.pad_input = nn.ZeroPad2d(
        #     conv2d_get_padding(
        #         x.shape[-2:],
        #         branch1.shape[-2:],
        #         kernel_size=3,
        #         stride=1,
        #         dilation=2,
        #     ))

        # padded_x = self.pad_input(x)
        # branch4 = self.branch4(padded_x)
        output = torch.cat([branch1, branch2, branch3], 1)
        output = self.conv(output)

        return output


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    num_classes, height, width = 4, 64, 512

    model = EvenBlock(num_classes)  # .to('cuda')
    inp = torch.randn(5, 1, height, width)  # .to('cuda')

    out = model(inp, inp)
    assert out.size() == torch.Size([5, num_classes, height, width])

    print("Pass size check.")
