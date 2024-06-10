import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
from skimage import io
import util.util as util
from . import random_crop_yh
import numpy as np
from collections import Counter
from . import local_hist

class valSegDataset(BaseDataset):
    def initialize(self, opt):
        print('**************yhSegDataset********************')
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.dir_A = opt.raw_MRI_dir
        self.dir_B = opt.raw_CT_dir
        self.dir_Seg = opt.raw_MRI_seg_dir

        self.A_paths = opt.imglist_MRI_val
        self.B_paths = opt.imglist_CT_val
        self.Seg_paths = opt.imglist_seg_val

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.Seg_size = len(self.Seg_paths)
        print('*&*&*&*&*&*^^^^',self.A_size,self.B_size,self.Seg_size)
        if not self.opt.isTrain:
            self.skipcrop = True
        else:
            self.skipcrop = False
        # self.transform = get_transform(opt)

        if self.skipcrop:
            osize = [opt.fineSize, opt.fineSize]
        else:
            osize = [opt.loadSize, opt.loadSize]
        transform_list = []
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Resize(osize, Image.NEAREST))
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_crop_yh.randomcrop_yh(opt.fineSize))
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize((0.5),
                                            (0.5)))
        self.transforms_normalize = transforms.Compose(transform_list)


        transform_list = []
        transform_list.append(transforms.Lambda(lambda img: transforms.functional.equalize(img)))
        self.transforms_HE = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.GaussianBlur(self.opt.gaussian_kernel_size, self.opt.gaussian_sigma))
        self.transforms_Gaussian = transforms.Compose(transform_list)

        # transform_list = []
        # transform_list.append(local_hist.LocalHistEqualization(self.opt.clip_limit))
        # self.transforms_LHE = transforms.Compose(transform_list)
        self.transforms_LHE = local_hist.LocalHistEqualization(self.opt.clip_limit)

    def print_pixel_counts(self, image, stage_identifier):
        """Utility function to print pixel counts and stage identifier"""
        image_num = np.asarray(image)
        pixel_values = image_num.ravel()
        pixel_counts = Counter(pixel_values)
        print(f"--- {stage_identifier} ---")
        for pixel_value, count in pixel_counts.items():
            print(f"Pixel value: {pixel_value}, Count: {count}")

    def __getitem__(self, index):

        index_A = index % self.A_size
        if self.opt.model == 'finetune':
            index_A = index % self.Seg_size


        A_path = self.A_paths[index_A]
        Seg_path = self.Seg_paths[index_A]
        # print('Seg path :  ',Seg_path)

        Seg_img = Image.open(Seg_path).convert('I')
        index_B = index_A
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('L')
        B_img = Image.open(B_path).convert('L')





        if self.opt.GaussianBlur:
            B_img = self.transforms_Gaussian(B_img)

        if self.opt.Local_Histogram_Equalization:
            slice_num = index_A%41
            if slice_num<20:
                x = self.opt.LHE_kernel_size_x
                y = self.opt.LHE_kernel_size_y
            else:
                x = self.opt.LHE_kernel_size_y
                y = self.opt.LHE_kernel_size_x

            B_img = self.transforms_LHE(B_img,x, y)

        # if self.opt.Histogram_Equalization:
        #     B_img = self.transforms_HE(B_img)


        A_img = self.transforms_scale(A_img)
        B_img = self.transforms_scale(B_img)
        Seg_img = self.transforms_seg_scale(Seg_img)


        if not self.skipcrop:
            [A_img,Seg_img] = self.transforms_crop([A_img, Seg_img])
            [B_img] = self.transforms_crop([B_img])


        A_img = self.transforms_toTensor(A_img)
        B_img = self.transforms_toTensor(B_img)
        Seg_img = self.transforms_toTensor(Seg_img)

        if self.opt.min_max_normalize:
            min_val_A = torch.min(A_img)
            max_val_A = torch.max(A_img)
            A_img = (A_img - min_val_A) / (max_val_A - min_val_A)

            min_val_B = torch.min(B_img)
            max_val_B = torch.max(B_img)
            B_img = (B_img - min_val_B) / (max_val_B - min_val_B)


        A_img = self.transforms_normalize(A_img)
        if self.opt.B_normalization:
            B_img = self.transforms_normalize(B_img)


        Seg_img[Seg_img != 200] = 0
        Seg_img[Seg_img == 200] = 1

        Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.fineSize, self.opt.fineSize)
        Seg_imgs[0, :, :] = Seg_img == 0
        Seg_imgs[1, :, :] = Seg_img == 1



        return {'A': A_img, 'B': B_img, 'Seg': Seg_imgs, 'Seg_one': Seg_img,
                'A_paths': A_path, 'B_paths': B_path, 'Seg_paths':Seg_path}


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
