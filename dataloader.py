import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path
from scipy import ndimage
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import glob
import random
from downloaded.routine import utils


def tensor_to_gpu(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


def tensor_to_cpu(tensor, is_cuda):
    if is_cuda:
        return tensor.cpu()
    else:
        return tensor

def train_transforms(imA, imB):
    im1 = TF.to_pil_image(imA)
    im2 = TF.to_pil_image(imB)
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(im1, output_size=(78, 78))
    im1 = TF.crop(im1, i, j, h, w)
    im2 = TF.crop(im2, i, j, h, w)
    # Random Vflip
    if random.random() > 0.5:
        im1 = TF.vflip(im1)
        im2 = TF.vflip(im2)
    # Random Hflip
    if random.random() > 0.5:
        im1 = TF.hflip(im1)
        im2 = TF.hflip(im2)
    # Random 90 degrees rotation
    if random.random() < 0.33:
        im1 = TF.rotate(im1,90)
        im2 = TF.rotate(im2,90)
    elif random.random() > 0.66:
        im1 = TF.rotate(im1,270)
        im2 = TF.rotate(im2,270)
    # To Tensor
    im1 = TF.to_tensor(im1) * 255
    im2 = TF.to_tensor(im2) * 255
    # Value normalization
    im1[0, :, :] = (im1[0, :, :]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im2[0, :, :] = (im2[0, :, :]*(235-16)+16)/255.0 #to [16/255, 235/255]
    return im1, im2

def test_transforms(imA, imB):
    im1 = TF.to_pil_image(imA)
    im2 = TF.to_pil_image(imB)
    # Center crop
    ccrop = transforms.CenterCrop(226)
    im1 = ccrop(im1)
    im2 = ccrop(im2)
    # To Tensor
    im1 = TF.to_tensor(im1) * 255
    im2 = TF.to_tensor(im2) * 255
    # Value normalization
    im1[0, :, :] = (im1[0, :, :]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im2[0, :, :] = (im2[0, :, :]*(235-16)+16)/255.0 #to [16/255, 235/255]
    return im1, im2

class SR_dataloader(Dataset):
    def __init__(self, Train=True, transform=None, dir='data/SR_training_datasets/T91', scale='2', use_cuda=False):
        self.transform = transform
        self.use_cuda = use_cuda
        self.Train = Train
        self.dir = dir
        self.sf = scale
        self.img_list = [os.path.basename(file) for file in glob.glob(dir+'/original/*.png')]
        self.img_num = len(self.img_list)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        hr_path = os.path.join(self.dir, 'x' + self.sf, self.img_list[idx])
        # read existing bicubic upscaled file (no need to upscale after reading)
        #bicub_path = os.path.join(self.dir, 'x' + self.sf, 'b_' + self.img_list[idx])
        # read LR file and upscale
        bicub_path = os.path.join(self.dir, 'x' + self.sf, 's_' + self.img_list[idx])

        hr_img = ndimage.imread(hr_path, mode="YCbCr")
        bicub_img = ndimage.imread(bicub_path, mode="YCbCr")
        bicub_img = utils.imresize(bicub_img, int(self.sf))

        if self.transform:
            hr_img, bicub_img = self.transform(hr_img, bicub_img)

        use_cuda = torch.cuda.is_available()
        hr_img = tensor_to_gpu(hr_img, use_cuda)
        bicub_img = tensor_to_gpu(bicub_img, use_cuda)

        return hr_img, bicub_img
