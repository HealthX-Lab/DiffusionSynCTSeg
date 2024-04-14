import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
from skimage import io
from . import random_crop_yh
import numpy as np
class yhSegDataset(BaseDataset):
    def initialize(self, opt):
        print('**************yhSegDataset********************')
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.dir_A = opt.raw_MRI_dir
        self.dir_B = opt.raw_CT_dir
        self.dir_Seg = opt.raw_MRI_seg_dir

        self.A_paths = opt.imglist_MRI
        self.B_paths = opt.imglist_CT
        self.Seg_paths = opt.imglist_seg

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
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


    def __getitem__(self, index):

        index_A = index % self.A_size

        # while 1:
        #    index_A = index % self.A_size
        #    Seg_path = self.Seg_paths[index_A]
        #    Seg_img = Image.open(Seg_path).convert('I')
        #    Seg_img = self.transforms_seg_scale(Seg_img)
        #    Seg_img = self.transforms_toTensor(Seg_img)
        #    Seg_img[Seg_img == 0] = 0
        #    Seg_img[Seg_img == 50] = 0
        #    Seg_img[Seg_img == 100] = 0
        #    Seg_img[Seg_img == 150] = 0
        #    Seg_img[Seg_img == 200] = 1
        #    print('while 1',index_A, Seg_path)
        #    if np.sum(Seg_img)>=1:
        #        print('while break', index_A, Seg_path)
        #        break
        A_path = self.A_paths[index_A]
        # Seg_path = A_path.replace(self.dir_A,self.dir_Seg)
        # Seg_path = Seg_path.replace('_rawimg','_organlabel')
        Seg_path = self.Seg_paths[index_A]
        # print('Seg path :  ',Seg_path)

        Seg_img = io.imread(Seg_path,as_gray=True)
        Seg_img = Image.fromarray(Seg_img)

        # print(f'new method for reading seg image min {np.min(Seg_img)}  and max {np.max(Seg_img)} hist   {hist}')
        #


        index_B = index_A
        B_path = self.B_paths[index_B]
        # print('## ',A_path,' ## ', B_path)
        A_img = Image.open(A_path).convert('L')
        # print('A_image',np.shape(A_img),flush=True)

        B_img = Image.open(B_path).convert('L')
        # print('B_image', np.shape(B_img),flush=True)

        A_img = self.transforms_scale(A_img)
        B_img = self.transforms_scale(B_img)

        Seg_img = self.transforms_seg_scale(Seg_img)


        if not self.skipcrop:
            [A_img,Seg_img] = self.transforms_crop([A_img, Seg_img])
            [B_img] = self.transforms_crop([B_img])

        # print('A_image_1', np.shape(A_img),flush=True)
        A_img = self.transforms_toTensor(A_img)
        # print('A_image_2', np.shape(A_img),flush=True)
        B_img = self.transforms_toTensor(B_img)
        Seg_img = self.transforms_toTensor(Seg_img)


        A_img = self.transforms_normalize(A_img)
        # print('A_image_3', np.shape(A_img), flush=True)
        # B_img = self.transforms_normalize(B_img)


        Seg_img[Seg_img == 0] = 0
        Seg_img[Seg_img == 1] = 1
        # Seg_num = Seg_img.numpy()
        #
        # # Compute histogram of pixel intensities
        # hist = np.histogram(Seg_num, bins=np.arange(np.min(Seg_num), np.max(Seg_num) + 2))
        #
        # print(f'min {np.min(Seg_num)}  max {np.max(Seg_num)}','',hist)

        Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.fineSize, self.opt.fineSize)
        Seg_imgs[0, :, :] = Seg_img == 0
        Seg_imgs[1, :, :] = Seg_img == 1


        return {'A': A_img, 'B': B_img, 'Seg': Seg_imgs, 'Seg_one': Seg_img,
                'A_paths': A_path, 'B_paths': B_path, 'Seg_paths':Seg_path}


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
