"""mobilenetv2 in pytorch
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):
    def __init__(self, class_num=100):
        super().__init__()
        self.class_num = class_num
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)     # in_channels, out_channels, stride, t
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)  # repeat, in_channels, out_channels, stride, t

        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)

        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)

        self.stage7 = LinearBottleNeck(160, 320, 1, 6)
        self.conv41 = nn.Sequential(nn.Conv2d(320, 1280, 1),
                                    nn.BatchNorm2d(1280),
                                    nn.ReLU6(inplace=True))
        self.conv42 = nn.Conv2d(1280, class_num, 1)

        self.conv11 = nn.Sequential(nn.Conv2d(320, 1280, 1),
                                    nn.BatchNorm2d(1280),
                                    nn.ReLU6(inplace=True))
        self.conv12 = nn.Conv2d(1280, class_num, 1)

        self.conv21 = nn.Sequential(nn.Conv2d(320, 1280, 1),
                                    nn.BatchNorm2d(1280),
                                    nn.ReLU6(inplace=True))
        self.conv22 = nn.Conv2d(1280, class_num, 1)

        self.conv31 = nn.Sequential(nn.Conv2d(320, 1280, 1),
                                    nn.BatchNorm2d(1280),
                                    nn.ReLU6(inplace=True))
        self.conv32 = nn.Conv2d(1280, class_num, 1)


        expansion = 1
        self.attention1 = nn.Sequential(LinearBottleNeck(in_channels=24 * expansion,
                                                         out_channels=24 * expansion, stride=1),
                                        nn.BatchNorm2d(24 * expansion),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor=1, mode='bilinear'),
                                        nn.Sigmoid())

        self.attention2 = nn.Sequential(LinearBottleNeck(in_channels=64 * expansion,
                                                         out_channels=64 * expansion, stride=1),
                                        nn.BatchNorm2d(64 * expansion),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor=1, mode='bilinear'),
                                        nn.Sigmoid())

        self.attention3 = nn.Sequential(LinearBottleNeck(in_channels=160 * expansion,
                                                         out_channels=160 * expansion, stride=1),
                                        nn.BatchNorm2d(160 * expansion),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor=1, mode='bilinear'),
                                        nn.Sigmoid())

        self.scala1 = nn.Sequential(LinearBottleNeck(in_channels=24 * expansion,
                                                     out_channels=64 * expansion, stride=1),
                                    LinearBottleNeck(in_channels=64 * expansion,
                                                     out_channels=160 * expansion, stride=1),
                                    LinearBottleNeck(in_channels=160 * expansion,
                                                     out_channels=320, stride=1))
        self.scala2 = nn.Sequential(LinearBottleNeck(in_channels=64 * expansion,
                                                     out_channels=160 * expansion, stride=1),
                                    LinearBottleNeck(in_channels=160 * expansion,
                                                     out_channels=320 * expansion, stride=1))
        self.scala3 = nn.Sequential(LinearBottleNeck(in_channels=160 * expansion,
                                                     out_channels=320 * expansion, stride=1),
                                    nn.AvgPool2d(4, 4))


    def forward(self, x):
        # not only FC, its combination of scala + attention
        feature_list = []

        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)         # After this FC1: torch.Size([2, 24, 33, 33])
        fea1 = self.attention1(x)  # torch.Size([2, 24, 33, 33])
        fea1 = fea1 * x
        feature_list.append(fea1)  # chanels 24 * expansion

        x = self.stage3(x)
        x = self.stage4(x)         # After this FC2: torch.Size([2, 64, 9, 9])
        fea2 = self.attention2(x)  # torch.Size([2, 64, 9, 9])
        fea2 = fea2 * x
        feature_list.append(fea2)  # chanels 64 * expansion

        x = self.stage5(x)
        x = self.stage6(x)         # After this FC3: torch.Size([2, 160, 9, 9])
        fea3 = self.attention3(x)  # torch.Size([2, 160, 9, 9])
        fea3 = fea3 * x
        feature_list.append(fea3)  # chanels 160 * expansion

        x = self.stage7(x)               # torch.Size([2, 320, 9, 9])
        x = self.conv41(x)               # torch.Size([2, 1280, 9, 9])
        x = F.adaptive_avg_pool2d(x, 1)  # torch.Size([2, 1280, 1, 1])
        feature_list.append(x)
        x = self.conv42(x)               # After this FC4: torch.Size([2, 100, 1, 1])

        # print("feature_list[0].shape", feature_list[0].shape)
        # print("feature_list[1].shape", feature_list[1].shape)
        # print("feature_list[2].shape", feature_list[2].shape)
        # print("feature_list[3].shape", feature_list[3].shape)

        feature1 = self.scala1(feature_list[0])
        feature1 = self.conv11(feature1)
        feature1 = F.adaptive_avg_pool2d(feature1, 1)
        out1_feature = feature1.view(x.size(0), -1)
        out1 = self.conv12(feature1).view(x.size(0), -1)

        feature2 = self.scala2(feature_list[1])
        feature2 = self.conv21(feature2)
        feature2 = F.adaptive_avg_pool2d(feature2, 1)
        out2_feature = feature2.view(x.size(0), -1)
        out2 = self.conv22(feature2).view(x.size(0), -1)

        feature3 = self.scala3(feature_list[2])
        feature3 = self.conv31(feature3)
        feature3 = F.adaptive_avg_pool2d(feature3, 1)
        out3_feature = feature3.view(x.size(0), -1)
        out3 = self.conv32(feature3).view(x.size(0), -1)


        # out4_feature = self.scala4(feature_list[3])
        out4_feature = feature_list[3].view(x.size(0), -1)
        out4 = x.view(x.size(0), -1)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)


def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)


if __name__ == "__main__":
    model = mobilenetv2(class_num=100)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print(y.shape)

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)