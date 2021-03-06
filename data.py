dataset.py                                                                                          0000644 0001750 0001750 00000014540 13673067134 011273  0                                                                                                    ustar   qyn                             qyn                                                                                                                                                                                                                    import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from skimage import img_as_float
from random import randrange
import os.path


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".jpg", ".jpeg"])


# def load_img(lrfilepath, htfilepath, nFrames):
#     if nFrames == 1:
#         input = Image.open(os.path.join(lrfilepath, 'im3.bmp')).convert('RGB')
#     else:
#         input = [Image.open(os.path.join(lrfilepath, 'im' + str(x) + '.png')).convert('RGB') for x in range(1, nFrames+1)]
#     target = Image.open(os.path.join(htfilepath, 'im3.bmp')).convert('RGB')
#     return input, target

def load_img(lrfilepath, htfilepath, nFrames):
    if nFrames == 1:
        input = Image.open(os.path.join('im3.bmp')).convert('RGB')
    else:
        input = [Image.open(os.path.join(lrfilepath+'/im' + str(x) + '.bmp')).convert('RGB') for x in range(1, nFrames+1)]
    target = Image.open(os.path.join(htfilepath, 'im3.bmp')).convert('RGB')
    return input, target

def get_patch(img_in, img_tar, patch_size, scale, nFrames, ix=-1, iy=-1):
    if nFrames == 1:
        (ih, iw) = img_in.size
    else:
        (ih, iw) = img_in[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)
    if nFrames == 1:
        img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    else:
        img_low = []
        for i in range(nFrames):
            img_low.append(img_in[i].crop((iy, ix, iy + ip, ix + ip)))

    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    if nFrames == 1:
        return img_in, img_tar, info_patch
    else:
        return img_low, img_tar, info_patch


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, file_list, nFrames, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir, file_list))]
        self.lr_image_filenames = [
            join(image_dir, 'trainlr', x) for x in alist]
        self.hr_image_filenames = [
            join(image_dir, 'trainhr', x) for x in alist]
        # self.lr_image_filenames = [
        #     join(image_dir, 'sequences', x) for x in alist]
        #
        # self.hr_image_filenames = [
        #     join(image_dir, 'sequences', x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size

    def __getitem__(self, index):
        input, target = load_img(self.lr_image_filenames[index], self.hr_image_filenames[index], self.nFrames)

        if self.patch_size != 0:
            input, target, _ = get_patch(input, target, self.patch_size, self.upscale_factor,
                                                  self.nFrames)

        if self.transform:
            # stack LQ images to NHWC, N is the frame number
            input = np.concatenate(input, axis=-1)
            # BGR to RGB, HWC to CHW, numpy to tensor
            target = self.transform(target)
            input = self.transform(input)

        return input, target, self.lr_image_filenames[index]

    def __len__(self):
        return len(self.lr_image_filenames)


class DatasetFromFolderValid(data.Dataset):
    def __init__(self, image_dir, file_list, nFrames, transform=None):
        super(DatasetFromFolderValid, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir, file_list))]
        self.image_filenames = [join(image_dir, x) for x in alist]
        self.lr_image_filenames = [
            join(image_dir, 'validlr', x) for x in alist]
        self.hr_image_filenames = [
            join(image_dir, 'validhr', x) for x in alist]
        self.nFrames = nFrames
        self.transform = transform

    def __getitem__(self, index):
        input, target = load_img(self.lr_image_filenames[index], self.hr_image_filenames[index], self.nFrames)

        if self.transform:
            # stack LQ images to NHWC, N is the frame number
            input = np.concatenate(input, axis=-1)
            # BGR to RGB, HWC to CHW, numpy to tensor
            target = self.transform(target)
            input = self.transform(input)

        return input, target, self.lr_image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, file_list, nFrames, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir, file_list))]
        self.image_filenames = [join(image_dir, x) for x in alist]
        self.lr_image_filenames = [
            join(image_dir, 'testlr', x) for x in alist]
        self.nFrames = nFrames
        self.transform = transform

    def __getitem__(self, index):
        input = [Image.open(os.path.join(self.lr_image_filenames[index], 'im' + str(x) + '.bmp')).convert('RGB') for x in
                 range(1, self.nFrames + 1)]

        if self.transform:
            # stack LQ images to NHWC, N is the frame number
            input = np.concatenate(input, axis=-1)
            # BGR to RGB, HWC to CHW, numpy to tensor
            input = self.transform(input)

        return input, self.lr_image_filenames[index]

    def __len__(self):
        return len(self.lr_image_filenames)

# input, target = load_img('./data/trainlr/00000/0000', './data/trainhr/00000/0000', 5)
# input, target, _ = get_patch(input, target, 56, 4, 5)
# train_set = DatasetFromFolder('./data', 'sep_trainlist.txt', 5, 56, 4, True)
# training_data_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=1, batch_size=1, shuffle=True)
#
# for batch in training_data_loader:
#     print(batch[0].data)                                                                                                                                                                main.py                                                                                             0000644 0001750 0001750 00000020236 13726664567 010605  0                                                                                                    ustar   qyn                             qyn                                                                                                                                                                                                                    import torch.nn as nn
import torch
from torch.autograd import Variable
import argparse
import torch.optim as optim
import cv2
import numpy as np
import pytorch_ssim

import os
from data import get_eval_set, get_training_set
from tensorboardX import SummaryWriter
import time


from model_VSR import DAVSR_1

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')


parser.add_argument('--dataDir', default='/home/qyn/qyn-prog/DASR/data', help='dataset directory')

parser.add_argument('--train_file_list', default='sep_trainlist.txt', help='dataset directory')
parser.add_argument('--eval_file_list', default='sep_validlist.txt', help='dataset directory')
parser.add_argument('--saveDir', default='./result', help='datasave directory')
parser.add_argument('--load', default='model_name', help='save result')

parser.add_argument('--pretrained', default=False, help='finetuning the training')


parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nChannel', type=int, default=15, help='number of channel')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')

parser.add_argument('--gamma', type=float, default=0.5, help='')

parser.add_argument('--patchSize', type=int, default=32, help='patch size')

parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--lrDecay', type=int, default=500, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lossType', default='L1', help='output SR video')
parser.add_argument('--nFrames', default=7, help='num Frames')
parser.add_argument('--scale', type=int, default=4, help='scale output size /input size')

parser.add_argument('--epochs', type=int, default=20000, help='number of epochs to train')
parser.add_argument('--model_name', default='_DAVSR_5', help='model to select')

args = parser.parse_args()
gpu_lists = [0]
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(123)
torch.cuda.manual_seed(123)

save_model_path = '/home/qyn/qyn-prog/DASR/models/' + args.model_name + '/'

print('===> Loading datasets')
eval_set = get_eval_set(args.dataDir, args.eval_file_list, 5)
eval_data_loader = torch.utils.data.DataLoader(dataset=eval_set, num_workers=args.nThreads, batch_size=1, shuffle=False)


train_set = get_training_set(args.dataDir, args.train_file_list, 5, args.patchSize, args.scale, True)

training_data_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=args.nThreads,
                                                   batch_size=args.batchSize,
                                                   shuffle=True)

model = DAVSR_5(args.nChannel, args.nFeat, args.scale)
model = nn.DataParallel(model, gpu_lists)
criterion = nn.L1Loss()


print('Pre-trained SR model is loaded.')

model = model.cuda()
criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

summarydir = '/home/qyn/qyn-prog/DASR/summary/' + args.model_name
writer = SummaryWriter(summarydir)


def test(epoch):
    from skimage.measure import compare_psnr
    from skimage import color
    avg_psnr = 0
    avg_mse = 0
    avg_ssim = 0
    model.eval()
    

    for batch in eval_data_loader:
        input, target = batch[0], batch[1]

        with torch.no_grad():
            input = Variable(input)
            target = Variable(target)

        input = input.cuda()
        target = target.cuda()
        mse = nn.MSELoss().cuda()

        with torch.no_grad():

            prediction = model(input)
          
            mse_loss = mse(prediction, target)

            ssim = pytorch_ssim.ssim(target, prediction)
            avg_ssim += ssim


            target = color.rgb2ycbcr(target.cpu().data.squeeze().numpy().transpose(1, 2, 0))[..., 0]
            prediction = color.rgb2ycbcr(prediction.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0))[..., 0]
            psnr = compare_psnr(target, prediction, 255.)
            avg_psnr += psnr
            avg_mse += mse_loss

           
    writer.add_scalar('avg_ssim', avg_ssim / len(eval_data_loader), global_step=epoch)
    writer.add_scalar('avg_psnr', avg_psnr / len(eval_data_loader), global_step=epoch)
    writer.add_scalar('avg_mse', avg_mse / len(eval_data_loader), global_step=epoch)

    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim / len(eval_data_loader)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(eval_data_loader)))
    print("===> Avg. MSE: {:.4f}".format(avg_mse / len(eval_data_loader)))


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
 
        t0 = time.time()
        input, target, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(input)

        mse_loss = criterion(prediction, target)
        loss = mse_loss
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        global_step = (epoch - 1) * len(training_data_loader) + iteration
        writer.add_scalar('total loss', loss, global_step=global_step)

        if iteration % 5 == 0:
            save_img(prediction[0].cpu().data, target[0].cpu().data, 1)

        print("===> Epoch[{}/{}] Step({}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, args.epochs,
                                                                                      global_step, loss.data,
                                                                                      (time.time() - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def save_img(img1, img2, epoch):

    save_img1 = img1.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_img2 = img2.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    saveimg = np.concatenate([save_img1, save_img2], 1)

    save_fn = str(epoch).zfill(6) + '.png'

    cv2.imwrite(save_fn, cv2.cvtColor(saveimg * 255., cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def checkpoint(epoch):

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    model_out_path = save_model_path + args.model_name + "_epoch_{}.pth".format(epoch)
    torch.save(model.module.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

'''
def checkpoint(epoch, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model_out_path = os.path.join(path, args.model_name + "_epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
'''

if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)

        # learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch + 1) % (args.epochs / 2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        if epoch % 100 == 0:
            checkpoint(epoch)
            test(epoch)
                                                                                                                                                                                                                                                                                                                                                                  model_VSR.py                                                                                        0000664 0001750 0001750 00000022250 13726665126 011503  0                                                                                                    ustar   qyn                             qyn                                                                                                                                                                                                                    import torch
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
                                                                                                                                                                                                                                                                                                                                                        test.py                                                                                             0000644 0001750 0001750 00000011061 13726665676 010636  0                                                                                                    ustar   qyn                             qyn                                                                                                                                                                                                                    from __future__ import print_function
import argparse

import os
import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import get_test_set
import skvideo.io
import numpy as np
import time
import cv2
import pytorch_ssim
from skimage.measure import compare_psnr

 from model_VSR import DAVSR_1


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=32, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=15, help='number of color channels to use')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# parser.add_argument('--data_dir', type=str, default='./data/')
parser.add_argument('--data_dir', type=str, default='/home/qyn/vimeo_super_resolution_test/low_resolution/')
parser.add_argument('--file_list', type=str, default='/home/qyn/vimeo_super_resolution_test/sep_testlist.txt')

parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')

parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')



parser.add_argument('--output', default='/media/qyn/Results1/DAVSR_1/', help='Location to save checkpoint models')


parser.add_argument('--model', default='/home/qyn/epochs_DAVSR_1/model_epoch_15000.pth',
                    help='sr pretrained base model')

parser.add_argument('--scale', type=int, default=4, help='scale output size /input size')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_test_set(opt.data_dir, opt.file_list, 5)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')

model = DAVSR_1(opt.nChannel, opt.nFeat, opt.scale)

device = torch.device("cuda")
model.load_state_dict(torch.load(opt.model))

model.to(device)

print('Pre-trained SR model is loaded.')


if cuda:
    model = model.cuda()



def test():
    model.eval()

 
    for batch in testing_data_loader:
        t0 = time.time()
        input, name = batch[0], batch[1]
        with torch.no_grad():
            input = Variable(input)
        if cuda:
            input = input.cuda()

        name_list = name[0].split('/')

        with torch.no_grad():
            prediction = model(input)
            ssim = pytorch_ssim.ssim(input, prediction)
            psnr = compare_psnr(input, prediction, 255.)
            save_img(prediction.cpu().data, name[0])
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (time.time() - t0)))


def save_feature_to_img(feature, path):
    #to numpy
    feature=feature.cpu().data.numpy()

    
    feature=np.round(feature*255)
    # print(feature[0])
    cv2.imwrite(path, feature)


def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    # save img
    save_dir = os.path.join(opt.output, img_name.split('/')[-2])
    name = str(int(img_name.split('/')[-1])+ 1).zfill(3)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir + '/' + 'Youku_{}_h_rdn_{}.bmp'.format(img_name.split('/')[-2], name)
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255., cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


test()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                               train.py                                                                                            0000644 0001750 0001750 00000017515 13560435475 010772  0                                                                                                    ustar   qyn                             qyn                                                                                                                                                                                                                    import torch.nn as nn
import torch
from torch.autograd import Variable
import argparse
import torch.optim as optim

from model import DASR
import os
from data import get_eval_set, get_training_set
from tensorboardX import SummaryWriter
import time

import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')

parser.add_argument('--dataDir', default='./data', help='dataset directory')
parser.add_argument('--train_file_list', default='sep_trainlist.txt', help='dataset directory')
parser.add_argument('--eval_file_list', default='sep_validlist.txt', help='dataset directory')
parser.add_argument('--saveDir', default='./result', help='datasave directory')
parser.add_argument('--load', default='model_name', help='save result')

parser.add_argument('--model_name', default='DASR', help='model to select')
parser.add_argument('--pretrained', default=False, help='finetuning the training')

parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nChannel', type=int, default=15, help='number of channel')

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
# parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
# parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
# parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')

parser.add_argument('--gamma', type=float, default=0.5, help='')

parser.add_argument('--patchSize', type=int, default=32, help='patch size')

parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=500, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lossType', default='L1', help='output SR video')
parser.add_argument('--nFrames', default=1, help='num Frames')
parser.add_argument('--scale', type=int, default=4, help='scale output size /input size')

args = parser.parse_args()
gpu_lists = [0]
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(123)
torch.cuda.manual_seed(123)

print('===> Loading datasets')
eval_set = get_eval_set(args.dataDir, args.eval_file_list, 5)
eval_data_loader = torch.utils.data.DataLoader(dataset=eval_set, num_workers=args.nThreads, batch_size=1, shuffle=False)
train_set = get_training_set(args.dataDir, args.train_file_list, 5, args.patchSize, args.scale, True)
training_data_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=args.nThreads,
                                                   batch_size=args.batchSize,
                                                   shuffle=True)

model = DASR(args)
# print(model)
model = nn.DataParallel(model, device_ids=gpu_lists)
criterion = nn.L1Loss()  # 两个网络的L1loss计算结构不一样？

# model.load_state_dict(torch.load('./models/DASR_epoch_140.pth'))
print('Pre-trained SR model is loaded.')

model = model.cuda()
criterion = criterion.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
writer = SummaryWriter('./summary/DASR')
''''
kwargs = {'map_location': lambda storage, loc: storage.cuda(0)}

def load_GPUS(model, model_path, kwargs):
    state_dict = torch.load(model_path, **kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        # name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model'''


# load_GPUS(model, './models/DASR_epoch_100.pth', kwargs)


def test(epoch):
    from skimage.measure import compare_psnr
    from skimage import color
    avg_psnr = 0
    for batch in eval_data_loader:
        # input, target = batch[1], batch[2]
        input, target = batch[0], batch[1]
        with torch.no_grad():
            input = Variable(input)
            target = Variable(target[0])
        input = input.cuda(gpu_lists[0])
        target = target.cuda(gpu_lists[0])

        with torch.no_grad():
            prediction = model(input)
        target = color.rgb2ycbcr(target.cpu().data.numpy().transpose(1, 2, 0))[..., 0]
        prediction = color.rgb2ycbcr(prediction.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0))[..., 0]
        psnr = compare_psnr(target, prediction, 255.)
        avg_psnr += psnr
    writer.add_scalar('avg_psnr', avg_psnr / len(eval_data_loader), global_step=epoch)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(eval_data_loader)))


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        t0 = time.time()
        input, target, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        input = input.cuda(gpu_lists[0])
        target = target.cuda(gpu_lists[0])

        optimizer.zero_grad()
        prediction = model(input)

        mse_loss = criterion(prediction, target)
        loss = mse_loss
        epoch_loss += loss.data

        loss.backward()
        optimizer.step()
        global_step = (epoch - 1) * len(training_data_loader) + iteration
        writer.add_scalar('total loss', loss, global_step=global_step)
        if iteration % 5 == 0:
            save_img(prediction[0].cpu().data, target[0].cpu().data, 1)

        print("===> Epoch[{}/{}] Step({}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, args.epochs,
                                                                                      global_step, loss.data,
                                                                                      (time.time() - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def save_img(img1, img2, epoch):
    save_img1 = img1.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_img2 = img2.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    # save img
    saveimg = np.concatenate([save_img1, save_img2], 1)

    save_fn = str(epoch).zfill(6) + '.png'

    cv2.imwrite(save_fn, cv2.cvtColor(saveimg * 255., cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def checkpoint(epoch):
    model_out_path = '/home/qyn/qyn-prog/DASR/models/' + args.model_name + "_epoch_{}.pth".format(epoch)
    torch.save(model.module.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
     for epoch in range(1, args.epochs + 1):
         train(epoch)

        # learning rate is decayed by a factor of 10 every half of total epochs
         if (epoch + 1) % (args.epochs / 2) == 0:
            for param_group in optimizer.param_groups:
                 param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

            if epoch % 10 == 0:
             checkpoint(epoch)
            # test(epoch)
    # test(100)                                                                                                                                                                                   utils.py                                                                                            0000644 0001750 0001750 00000002240 13556642276 011006  0                                                                                                    ustar   qyn                             qyn                                                                                                                                                                                                                    import os
import os.path
import torch


class saveData():
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.join(args.saveDir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def save_model(self, model):
        torch.save(
            model.state_dict(),
            self.save_dir_model + '/model_lastest.pt')
        torch.save(
            model,
            self.save_dir_model + '/model_obj.pt')

    def save_log(self, log):
        self.logFile.write(log + '\n')

    def load_model(self, model):
        model.load_state_dict(torch.load(self.save_dir_model + '/model_lastest.pt'))
        print("load mode_status frmo {}/model_lastest.pt".format(self.save_dir_model))
        return model
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                