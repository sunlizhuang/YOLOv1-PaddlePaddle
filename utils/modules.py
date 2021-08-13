import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Hardswish(nn.Layer):
    @staticmethod
    def forward(x):
        return x * F.relu6(x + 3.0) / 6.0


class Conv(nn.Layer):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, leaky=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2D(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2D(c2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.convs(x)

