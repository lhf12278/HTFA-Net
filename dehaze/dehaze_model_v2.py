import torch.nn as nn
import torch


class one_conv(nn.Module):
    def __init__(self, in_chanels, G, kernel_size=3):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(in_chanels, G, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), 1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class Channel_Attention(nn.Module):
    def __init__(self, in_planes=64, ratio=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class Position_Attention(nn.Module):
    def __init__(self):
        super(Position_Attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CPA(nn.Module):
    def __init__(self):
        super(CPA, self).__init__()
        self.ca = Channel_Attention()
        self.pa = Position_Attention()

    def forward(self, x):
        ca_out = self.ca(x) * x
        cpa_out = self.pa(ca_out) * ca_out
        return cpa_out


class ADB(nn.Module):
    def __init__(self, G0, C, G):
        super(ADB, self).__init__()
        self.rdb_0 = RDB(G0, C, G)
        self.cpa = CPA()
        self.rdb_1 = RDB(G0, C, G)

    def forward(self, x):
        out = self.rdb_0(x)
        cpa_out = self.cpa(out)
        ADB_out = self.rdb_1(cpa_out)
        return ADB_out


class featrue_separat(nn.Module):
    def __init__(self, G0, G, kernel_size=3):
        super(featrue_separat, self).__init__()
        self.separ = nn.Sequential(
            nn.Conv2d(G0, G0 * 2, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(G0 * 2, G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),

            nn.ReLU(),
            nn.Conv2d(G0, G0, kernel_size=1, padding=1 >> 1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        M = self.separ(x)
        E = torch.ones_like(M)
        m = E - M
        return M, m


class detail_add(nn.Module):
    def __init__(self, G0, kernel_size=3):
        super(detail_add, self).__init__()
        self.detail_extraction = nn.Sequential(
            nn.Conv2d(G0, G0 * 2, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(G0 * 2, G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
        )

    def forward(self, x):
        f3 = self.detail_extraction(x)
        return f3


def horizontalslice(x):
    height = x.shape[2]
    h_up = height // 2
    h_d = h_up + 1
    upslice = x[:, :, 0:h_up, :]
    downslice = x[:, :, h_up:height, :]
    return upslice, downslice


def verticalslice(x):
    weight = x.shape[3]
    w_left = weight // 2
    leftslice = x[:, :, :, 0:w_left]
    rightslice = x[:, :, :, w_left:weight]
    return leftslice, rightslice


def vhslice(x):
    upslice, downslice = horizontalslice(x)
    leftslice_1, rightslice_1 = verticalslice(upslice)
    leftslice_2, rightslice_2 = verticalslice(downslice)
    return leftslice_1, rightslice_1, leftslice_2, rightslice_2


class separat_block(nn.Module):
    def __init__(self, G0, C, G):
        super(separat_block, self).__init__()

        self.rdb = RDB(G0, C, G)
        self.cpa = CPA()

    def forward(self, x):
        F22_1_ = self.rdb(x)
        F22_1 = self.cpa(F22_1_)
        upslice1, downslice1 = horizontalslice(x)
        F22_20_ = self.rdb(upslice1)
        F22_20 = self.cpa(F22_20_)
        F22_21_ = self.rdb(downslice1)
        F22_21 = self.cpa(F22_21_)
        F22_2 = torch.cat((F22_20, F22_21), 2)

        leftslice1, rightslice1 = verticalslice(x)
        F22_30_ = self.rdb(leftslice1)
        F22_30 = self.cpa(F22_30_)
        F22_31_ = self.rdb(rightslice1)
        F22_31 = self.cpa(F22_31_)
        F22_3 = torch.cat((F22_30, F22_31), 3)

        leftslice_11, rightslice_11, leftslice_21, rightslice_21 = vhslice(x)
        F22_40_ = self.rdb(leftslice_11)
        F22_40 = self.cpa(F22_40_)
        F22_41_ = self.rdb(rightslice_11)
        F22_41 = self.cpa(F22_41_)
        F22_42_ = self.rdb(leftslice_21)
        F22_42 = self.cpa(F22_42_)
        F22_43_ = self.rdb(rightslice_21)
        F22_43 = self.cpa(F22_43_)

        F22_44 = torch.cat((F22_40, F22_41), 2)
        F22_45 = torch.cat((F22_42, F22_43), 2)
        F22_4 = torch.cat((F22_44, F22_45), 3)
        F22_total = F22_1 + F22_2 + F22_3 + F22_4
        return F22_total


class PAM_depth2shallow(nn.Module):
    def __init__(self):
        super(PAM_depth2shallow, self).__init__()
        self.deline = nn.Conv2d(64, 32, kernel_size=1, padding=1 >> 1, stride=1)
        self.max = nn.MaxPool2d(4, 4)
        self.dconv = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.increase = nn.Conv2d(32, 64, kernel_size=1, padding=1 >> 1, stride=1)
        self.fusion0 = nn.Conv2d(96, 64, kernel_size=1, padding=1 >> 1, stride=1)

    def forward(self, x, F_depth):
        F_depth1 = self.deline(F_depth)
        F_max = self.max(F_depth1)
        x_max = self.max(x)
        batchsize, C, w, h = x_max.size()
        proj_query = F_max.view(batchsize, -1, w * h).permute(0, 2, 1)
        proj_key = x_max.view(batchsize, -1, w * h)
        torch.cuda.empty_cache()
        attention = self.softmax(torch.bmm(proj_query, proj_key))

        proj_value = x_max.view(batchsize, -1, w * h)

        a = torch.bmm(proj_value, attention.permute(0, 2, 1))
        a = a.view(batchsize, C, w, h)

        a = self.dconv(a)
        F_strength_ = self.gamma * a

        F_strength0 = self.fusion0(torch.cat((x, F_depth1, F_strength_), 1))

        return F_strength0


class PAM_shallow2depth(nn.Module):
    def __init__(self):
        super(PAM_shallow2depth, self).__init__()
        self.deline = nn.Conv2d(64, 32, kernel_size=1, padding=1 >> 1, stride=1)
        self.max = nn.MaxPool2d(4, 4)
        self.dconv = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.increase = nn.Conv2d(32, 64, kernel_size=1, padding=1 >> 1, stride=1)
        self.fusion1 = nn.Conv2d(96, 64, kernel_size=1, padding=1 >> 1, stride=1)

    def forward(self, x, F_depth):
        F_depth0 = self.deline(F_depth)
        F_max = self.max(F_depth0)
        x_max = self.max(x)
        batchsize, C, w, h = x_max.size()
        proj_query = x_max.view(batchsize, -1, w * h).permute(0, 2, 1)
        proj_key = F_max.view(batchsize, -1, w * h)

        torch.cuda.empty_cache()
        attention = self.softmax(torch.bmm(proj_query, proj_key))
        proj_value = F_max.view(batchsize, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, w, h)
        out = self.dconv(out)

        F_strength_0 = self.gamma * out

        F_strength2 = self.fusion1(torch.cat((x, F_depth0, F_strength_0), 1))

        return F_strength2


class dehaze_network(nn.Module):
    def __init__(self, args):
        super(dehaze_network, self).__init__()
        self.D = args.D
        self.C = args.C
        self.G = args.G
        self.G0 = args.G0
        self.D_dehaze = args.D_dehaze
        print("D:{},C:{},G:{},G0:{}".format(self.D, self.C, self.G, self.G0))
        kernel_size = args.kernel_size
        input_channels = args.input_channels
        out_channels = args.out_channels
        # shallow feature extraction
        self.SFE1 = nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.SFE2 = nn.Conv2d(32, self.G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)

        ######################################################################
        self.fs = featrue_separat(self.G0, self.G)
        self.da = detail_add(self.G0)
        self.fb = separat_block(self.G0, self.C, self.G)
        ######################################################################

        self.ADBS = nn.ModuleList()
        for d in range(self.D_dehaze):
            self.ADBS.append(ADB(self.G0, self.C, self.G))
        self.depth2shallow = PAM_depth2shallow()
        self.shallow2depth = PAM_shallow2depth()

        self.GFF = nn.Sequential(
            nn.Conv2d(self.G0, self.G0, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.G0, self.G0, kernel_size, padding=kernel_size >> 1, stride=1),
            nn.Tanh()
        )
        self.recons = nn.Sequential(
            nn.Conv2d(self.G0, self.G0*2, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.G0*2, self.G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
            nn.Conv2d(self.G0, out_channels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        f__1 = self.SFE1(x)
        out = self.SFE2(f__1)
        F_out = out
        ADB_outs = []
        for i in range(self.D_dehaze):
            out = self.ADBS[i](out)
            ADB_outs.append(out)

        out1 = torch.zeros_like(ADB_outs[0])
        for k in range(len(ADB_outs)):
            a = ADB_outs[k]
            out1 = out1 + ADB_outs[k]
            k = k + 1
        F_depth = out1
        F_strength0 = self.depth2shallow(f__1, F_depth)
        F_strength1 = self.shallow2depth(f__1, F_depth)
        F11 = F_strength0 + F_strength1
        out1 = F11
        F22 = self.GFF(out1)
        out_Img = self.recons(F22)

        return F11, F22, out_Img


