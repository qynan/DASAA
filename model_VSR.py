import torch
import cv2
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F
from torch.autograd import Variable
from skimage import morphology

from thop import profile,clever_format

class myLSTM(nn.Module):
    def __init__(self, nFeat):
        super(myLSTM, self).__init__()
        self.conv_i = nn.Sequential(
            nn.Conv2d(nFeat + nFeat, nFeat, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(nFeat + nFeat, nFeat, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(nFeat + nFeat, nFeat, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(nFeat + nFeat, nFeat, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        
        h = Variable(torch.zeros(batch_size, 64, row, col))
        c = Variable(torch.zeros(batch_size, 64, row, col))

        h = h.cuda()
        c = c.cuda()

        x = torch.cat((input, h), 1)
        f = self.conv_f(x)
        i = self.conv_i(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = c * f + i * g
        h = o * F.tanh(c)
        return h

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_avg = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
        )
        self.conv_max = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y3 = self.conv_avg(y1)
        y4 = self.conv_max(y2)
        out1 = y3 + y4
        out2 = self.sigmoid(out1)
        out = out2 * x

        return out


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()

        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_p = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
        )
        self.conv_c = nn.Conv2d(2*channel, channel, kernel_size=1, padding=0, bias=True)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y3 = self.conv_p(y1)
        y4 = self.conv_p(y2)
        y5 = y1+y3
        y6 = y2+y4
        out_cat = torch.cat([y5, y6], dim=1)
        out_cat = self.conv_c(out_cat)
       
        out_s = self.sigmoid(out_cat)
        out = out_s * x
        return out

class DAM2(nn.Module):
    def __init__(self, nFeat, in_dim, channels, reduction):
        super(DAM2, self).__init__()
        self.in_dim = nFeat
        self.channels = nFeat

        self.conv = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nFeat, nFeat, 3, padding=1, bias=True)
        )
        self.pal = PALayer(in_dim)
        self.cal = CALayer(in_dim, reduction=16)

        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
      
        self.fusion = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)

    def __call__(self, x_p, x_c, is_training=1):
        
        b, c, h, w = x_p.shape
        buffer_p = self.pal(x_p)
        buffer_c = self.cal(x_c)

        # M{channel to position}
        Q1 = self.b1(buffer_p).permute(0, 2, 3, 1)  # B*H*W*C
        S1 = self.b2(buffer_c).permute(0, 2, 1, 3)  # B*H*C*W
        score1 = torch.matmul(Q1.contiguous().view(-1, w, c),
                              S1.contiguous().view(-1, c, w))  # B*H*W*W
        M_c_to_p = self.softmax(score1)

        # M{position to channel}
        Q2 = self.b1(buffer_c).permute(0, 2, 3, 1)  # B*H*W*C
        S2 = self.b2(buffer_p).permute(0, 2, 1, 3)  # B*H*W*C
        score2 = torch.matmul(Q2.contiguous().view(-1, w, c),
                              S2.contiguous().view(-1, c, w))  # B*H*W*W
        M_p_to_c = self.softmax(score2)

        # mask
        V_p_to_c = torch.sum(M_p_to_c.detach(), 1) > 0.1
        V_p_to_c = V_p_to_c.view(b, 1, h, w)  # B * 1 * H * W
        V_p_to_c = morphologic_process(V_p_to_c)
        if is_training == 1:
            V_c_to_p = torch.sum(M_c_to_p.detach(), 1) > 0.1
            V_c_to_p = V_c_to_p.view(b, 1, h, w)  # B * 1 * H * W
            V_c_to_p = morphologic_process(V_c_to_p)

            M_p_c_p = torch.bmm(M_c_to_p, M_p_to_c)
            M_c_p_c = torch.bmm(M_p_to_c, M_c_to_p)

        # fusion
        buffer1 = self.b3(x_c).permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer = torch.matmul(M_c_to_p, buffer1).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W
        out = self.fusion(torch.cat((buffer, x_p, V_p_to_c), 1))

        return out


def morphologic_process(mask):
    device = mask.device
    b, _, _, _ = mask.shape
    mask = 1 - mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx, 0, :, :], ((3, 3), (3, 3)), 'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx, 0, :, :] = buffer[3:-3, 3:-3]
    mask_np = 1 - mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)


class sub_pixel(nn.Module):
    def __init__(self, scale):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class DAVSR_1(nn.Module):
    def __init__(self, Channel, nfeat, scale):
        super(DAVSR_1, self).__init__()
        self.conv1 = nn.Conv2d(Channel, nfeat, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nfeat, nfeat, 3, padding=1, bias=True)
        self.RDB1 = RDB(nChannels=nfeat, nDenselayer=6, growthRate=32)
        self.RDB2 = RDB(nChannels=nfeat, nDenselayer=6, growthRate=32)
        self.RDB3 = RDB(nChannels=nfeat, nDenselayer=6, growthRate=32)
        self.mylstm = myLSTM(nFeat=nfeat)
        self.dam = DAM2(nFeat=nfeat, in_dim=nfeat, channels=nfeat, reduction=16)
        # Upsampler
        self.conv_up = nn.Conv2d(nfeat, nfeat * scale * scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        
       
        self.conv3 = nn.Conv2d(nfeat, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.RDB1(feat2)
        feat4 = self.RDB2(feat3)
        feat5 = self.RDB3(feat4)
        feat6 = self.dam(feat5, feat5)
        us1 = self.conv_up(feat6)
        us2 = self.upsample(us1)
        feat7 = self.mylstm(us2)
        out = self.conv3(feat7)
        return out


if __name__ == '__main__':
    x = torch.rand(1, 15, 32, 32)
    model = DAVSR_1(15, 64, 4)

    y = model(x)
    print(y, y.shape)

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
