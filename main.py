import torch.nn as nn
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
