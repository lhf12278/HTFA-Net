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


class PAM_depth2shallow(nn.Module):  
    def __init__(self):
        super(PAM_depth2shallow, self).__init__()
       
        self.deline = nn.Conv2d(64, 32, kernel_size=1, padding=1 >> 1, stride=1)
        
        self.max = nn.MaxPool2d(4, 4)
        self.dconv = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.increase = nn.Conv2d(32, 64, kernel_size=1, padding=1 >> 1, stride=1)
        self.fusion0 = nn.Conv2d(192, 64, kernel_size=1, padding=1 >> 1, stride=1)

    def forward(self, x, F_depth):
        F_max = self.max(F_depth)   
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
        a = self.gamma * a + x   
        F_strength_ = a + x
        
        F_strength0 = self.fusion0(torch.cat((x, F_depth, F_strength_), 1))

        return F_strength0


class PAM_shallow2depth(nn.Module):  
    def __init__(self):
        super(PAM_shallow2depth, self).__init__()
        self.deline = nn.Conv2d(64, 32, kernel_size=1, padding=1 >> 1, stride=1)
        self.max = nn.MaxPool2d(4, 4)
        self.dconv = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.increase = nn.Conv2d(32, 64, kernel_size=1, padding=1 >> 1, stride=1)
        self.fusion1 = nn.Conv2d(192, 64, kernel_size=1, padding=1 >> 1, stride=1)

    def forward(self, x, F_depth):
        F_max = self.max(F_depth)   
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

        F_strength_0 = self.gamma * out + x

        F_strength2 = self.fusion1(torch.cat((x, F_depth, F_strength_0), 1))

        return F_strength2


class Feature_extractor_reconstruction(nn.Module):
    def __init__(self, args):
        super(Feature_extractor_reconstruction, self).__init__()
        self.D = args.D
        self.C = args.C
        self.G = args.G
        self.G0 = args.G0
        print("D:{},C:{},G:{},G0:{}".format(self.D, self.C, self.G, self.G0))
        kernel_size = args.kernel_size
        input_channels = args.input_channels
        out_channels = args.out_channels
       
        self.SFE1 = nn.Conv2d(input_channels, self.G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.SFE2 = nn.Conv2d(self.G0, self.G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        
        self.ADBS = nn.ModuleList()
        for d in range(self.D):
            self.ADBS.append(ADB(self.G0, self.C, self.G))
        self.depth2shallow = PAM_depth2shallow()
        self.shallow2depth = PAM_shallow2depth()
        
        self.GFF = nn.Sequential(
            nn.Conv2d(self.D * self.G0, self.G0, kernel_size=1, padding=0, stride=1), 
            nn.Conv2d(self.G0, self.G0, kernel_size, padding=kernel_size >> 1, stride=1), 
        )
        self.recons = nn.Sequential(
            nn.Conv2d(self.G0, self.G0*2, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.G0*2, self.G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.G0, out_channels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
        )

    def forward(self, x):
        f__1 = self.SFE1(x)  
        f__2 = self.SFE2(f__1)
        ADB_outs = []
        for i in range(self.D):
            out = self.ADBS[i](f__2)
            ADB_outs.append(out)
            if i == 4:
                F_depth = out
                F_strength0 = self.depth2shallow(f__1, F_depth)
                F_strength1 = self.shallow2depth(f__1, F_depth)
                F1 = F_strength0 + F_strength1
                out = F1

        out = torch.cat(ADB_outs, 1)
        out = self.GFF(out)
        
        F2 = out
        out_Img = self.recons(F2)
        return F1, F2, out_Img





