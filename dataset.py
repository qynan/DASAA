import torch.utils.data as data
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
#     print(batch[0].data)