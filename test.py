from __future__ import print_function
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

