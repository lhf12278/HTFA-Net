import functools
import torch.nn as nn
import torch
import torch.nn.functional


class Trans_fineNet(nn.Module):
    def __init__(self):
      super(Trans_fineNet, self).__init__()
      Trans_coarseNet_conv_1 = nn.Conv2d(in_channels=3,
                             out_channels=5,
                             kernel_size=11,
                             stride=1,
                             padding=11>>1,
                             bias=True)
      Trans_coarseNet_conv_2 = nn.Conv2d(5, 5, 9, 1, 9 >> 1, bias=True)
      Trans_coarseNet_conv_3 = nn.Conv2d(5, 10, 7, 1, 7 >> 1, bias=True)
      Trans_coarseNet_conv_4 = nn.Conv2d(10, 1, 1, 1, padding=1 >> 1, bias=True)
      Trans_coarseNet_Conv = [Trans_coarseNet_conv_1, Trans_coarseNet_conv_2, Trans_coarseNet_conv_3, Trans_coarseNet_conv_4]

      # ###############################################################
      self.condition_conv = nn.Sequential(*Trans_coarseNet_Conv)
      self.conv1 = nn.Conv2d(in_channels=3,
                             out_channels=4,
                             kernel_size=7,
                             stride=1,
                             padding=7 >> 1,
                             bias=True)
      self.conv2 = nn.Conv2d(5, 5, 5, 1, 5 >> 1, bias=True)
      self.conv3 = nn.Conv2d(5, 10, 3, 1, 3 >> 1, bias=True)
      self.conv4 = nn.Conv2d(10, 1, 1, 1, 1 >> 1, bias=True)

    def forward(self, x):
        F0 = self.condition_conv(x)
        coarse_transMap = torch.sigmoid(F0)
        F_ = self.conv1(x)
        F = torch.cat((F_, coarse_transMap), 1)
        F_0 = self.conv2(F)
        F_1 = self.conv3(F_0)

        fine_transMap = torch.sigmoid(self.conv4(F_1))
        return fine_transMap


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
             use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
             use_bias = norm_layer == nn.InstanceNorm2d

        self.layer_1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(ndf * 1, ndf * 2, kernel_size=4, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 2, kernel_size=4, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=1, padding=2),
            nn.Sigmoid()
        )

    def forward(self, input):
        feature_1 = self.layer_1(input)
        feature_2 = self.layer_2(feature_1)
        feature_3 = self.layer_3(feature_2)
        feature_4 = self.layer_4(feature_3)
        result = self.layer_5(feature_4)
        return feature_2, feature_4, result


class _FeatureBlockDiscriminator2(nn.Module):
    def __init__(self, input_nc, w, h):
        super(_FeatureBlockDiscriminator2, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Linear(166400, input_nc),
            nn.PReLU(),
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(input_nc, input_nc),
            nn.PReLU(),
            nn.Sigmoid()
        )

        self.layer_3 = nn.Linear(input_nc, 2)

    def forward(self, x2):
        result2 = x2.contiguous().view(-1, x2.size(1) * x2.size(2) * x2.size(3))  # 306432
        layer1 = self.layer_1(result2)
        layer2 = self.layer_2(layer1)
        output = self.layer_3(layer2)
        return output


class _FeatureBlockDiscriminator4(nn.Module):
    def __init__(self, input_nc, w, h):
        super(_FeatureBlockDiscriminator4, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Linear(306432, input_nc),
            nn.PReLU(),
            nn.Sigmoid()
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(input_nc, input_nc),
            nn.PReLU(),

        )

        self.layer_3 = nn.Linear(input_nc, 2)

    def forward(self, x4):
        result4 = x4.contiguous().view(-1, x4.size(1) * x4.size(2) * x4.size(3))  # 306432
        layer1 = self.layer_1(result4)
        layer2 = self.layer_2(layer1)
        output = self.layer_3(layer2)

        return output