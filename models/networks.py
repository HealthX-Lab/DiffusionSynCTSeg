import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from . import FCNGCN
import math
from torch import Tensor, einsum
from typing import List

import timm
import torch
import torch.nn as nn


class SwinTinyForSegmentation(nn.Module):
    def __init__(self, input_nc, output_nc, enable_dropout=True, img_size=224, pretrained=False,  dropout_rate=0.2):
        super().__init__()
        # Load the Swin Transformer Tiny model
        model_name = 'swin_tiny_patch4_window7_224'
        self.swin_tiny = timm.create_model(model_name, pretrained=pretrained, in_chans=input_nc)


        # Include dropout if enable_dropout is True
        self.enable_dropout = enable_dropout
        self.dropout = nn.Dropout2d(dropout_rate) if enable_dropout else None

        # Segmentation head with output_nc channels
        # self.segmentation_head = nn.Conv2d(self.swin_tiny.num_features, output_nc, kernel_size=1)
        self.segmentation_head  = nn.ConvTranspose2d(self.swin_tiny.num_features, output_nc, kernel_size=32, stride=32)


        # Image size
        self.img_size = img_size

    def forward_features(self, x):
        x = self.swin_tiny.forward_features(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        if self.enable_dropout:
            x = self.dropout(x)
        print('swin shape', x.shape)
        x = self.segmentation_head(x)
        print('head shape', x.shape)
        # Upsample the output to the desired image size
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear',
                                                align_corners=False)
        return x
###############################################################################
# Functions
###############################################################################
class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        # assert simplex(probs)
        # assert not one_hot(dist_maps)
        print('****',probs.shape, dist_maps.shape)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


def mutual_information(hgram):
    """ Compute mutual information for the given joint histogram """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def calculate_mutual_information(img1, img2, bins=20):
    """ Calculate mutual information between two images """
    # Check if images are torch tensors and convert them to numpy arrays
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach().numpy()

    # Flatten the images
    img1_np = img1.flatten()
    img2_np = img2.flatten()

    # Compute the joint histogram
    joint_histogram = np.histogram2d(img1_np, img2_np, bins=bins)[0]

    # Compute the mutual information
    mi = mutual_information(joint_histogram)

    return mi

def Cor_CoeLoss(y_pred, y_target):
    x = y_pred
    y = y_target
    x_var = x - torch.mean(x)
    y_var = y - torch.mean(y)
    r_num = torch.sum(x_var * y_var)
    r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
    r = r_num / r_den

    # return 1 - r  # best are 0
    return 1 - r**2    # abslute constrain


class MIND(torch.nn.Module):
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MIND, self).__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma2 = gaussian_patch_sigma * gaussian_patch_sigma

        # calc shifted images in non local region
        self.image_shifter = torch.nn.Conv2d(in_channels=1, out_channels=self.nl_size * self.nl_size,
                                             kernel_size=(self.nl_size, self.nl_size),
                                             stride=1, padding=((self.nl_size-1)//2, (self.nl_size-1)//2),
                                             dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            t = torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i % self.nl_size, i//self.nl_size] = 1
            self.image_shifter.weight.data[i] = t

        # patch summation
        self.summation_patcher = torch.nn.Conv2d(in_channels=self.nl_size*self.nl_size, out_channels=self.nl_size*self.nl_size,
                                                 kernel_size=(self.p_size, self.p_size),
                                                 stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                                 dilation=1, groups=self.nl_size*self.nl_size, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            # gaussian kernel
            t = torch.zeros((1, self.p_size, self.p_size))
            cx = (self.p_size-1)//2
            cy = (self.p_size-1)//2
            for j in range(self.p_size * self.p_size):
                x = j % self.p_size
                y = j//self.p_size
                d2 = torch.norm(torch.tensor([x-cx, y-cy]).float(), 2)
                t[0, x, y] = math.exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] = t

        # neighbor images
        self.neighbors = torch.nn.Conv2d(in_channels=1, out_channels=self.n_size*self.n_size,
                                         kernel_size=(self.n_size, self.n_size),
                                         stride=1, padding=((self.n_size-1)//2, (self.n_size-1)//2),
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.zeros((1, self.n_size, self.n_size))
            t[0, i % self.n_size, i//self.n_size] = 1
            self.neighbors.weight.data[i] = t


        # neighbor patcher
        self.neighbor_summation_patcher = torch.nn.Conv2d(in_channels=self.n_size*self.n_size, out_channels=self.n_size * self.n_size,
                                                          kernel_size=(self.p_size, self.p_size),
                                                          stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                                          dilation=1, groups=self.n_size*self.n_size, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] = t

    def forward(self, orig):
        assert(len(orig.shape) == 4)
        assert(orig.shape[1] == 1)

        # get original image channel stack
        orig_stack = torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size*self.nl_size)], dim=1)

        # get shifted images
        shifted = self.image_shifter(orig)

        # get image diff
        diff_images = shifted - orig_stack

        # diff's L2 norm
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))

        # calc neighbor's variance
        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))
        Vx = neighbor_images.var(dim=1).unsqueeze(dim=1)

        # output mind
        nume = torch.exp(-Dx_alpha / (Vx + 1e-8))
        denomi = nume.sum(dim=1).unsqueeze(dim=1)
        mind = nume / denomi
        return mind


class MINDLoss(torch.nn.Module):
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0,loss_type='normal'):
        super(MINDLoss, self).__init__()
        self.nl_size = non_local_region_size
        self.loss_type = loss_type
        self.MIND = MIND(non_local_region_size=non_local_region_size,
                         patch_size=patch_size,
                         neighbor_size=neighbor_size,
                         gaussian_patch_sigma=gaussian_patch_sigma)

    def forward(self, input, target):
        in_mind = self.MIND(input)
        tar_mind = self.MIND(target)
        loss = 0
        if self.loss_type=='normal':
            mind_diff = in_mind - tar_mind
            l1 =torch.norm(mind_diff, 1)
            loss = l1 / (input.shape[2] * input.shape[3] * self.nl_size * self.nl_size)
        else:
            loss = Cor_CoeLoss(in_mind, tar_mind)
        return loss



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=True, gpu_ids=[],uncertainty=False,leaky_relu = False,seg=False, mode='2d'):
    netG = None
    print('**which_model_netG, use_dropout',which_model_netG, use_dropout)
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids,uncertainty= uncertainty,leaky_relu=leaky_relu,seg=seg)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids, uncertainty=uncertainty,leaky_relu=leaky_relu,seg=seg)
    elif which_model_netG == 'unet_128' or which_model_netG == 'unet_256':
        # netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        netG = U_Net(input_nc, output_nc,enable_dropout= use_dropout, mode=mode)
    elif which_model_netG == 'R2AttU_Net':
        netG = R2AttU_Net(input_nc, output_nc,enable_dropout= use_dropout, mode=mode)
    elif which_model_netG == 'AttU_Net':
        netG = AttU_Net(input_nc, output_nc,enable_dropout= use_dropout, mode=mode)
    elif which_model_netG == 'R2U_Net':
        netG = R2U_Net(input_nc, output_nc,enable_dropout= use_dropout, mode=mode)
    elif which_model_netG == 'swin':
        netG =  SwinTinyForSegmentation(input_nc, output_nc,enable_dropout= use_dropout)
    elif which_model_netG == 'seg_GCN_50':
        netG = FCNGCN.FCNGCN(num_input_chanel=input_nc, num_classes=output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=True, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            print('************MSEloss**************')
            self.loss = nn.MSELoss()
        else:
            print('************BCEloss**************')
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()
        # print('in WassersteinLoss')


    def __call__(self, real_logits=[], generated_logits=[], discriminator=True):
        if discriminator:
            real_loss = -torch.mean(real_logits)
            generated_loss = torch.mean(generated_logits)
            discriminator_loss = real_loss + generated_loss
            # print('WassersteinLoss discriminator ', discriminator_loss)
            return discriminator_loss
        elif not discriminator:
            generator_loss = -torch.mean(generated_logits)
            # print('WassersteinLoss generator ', generator_loss)
            return generator_loss






# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],uncertainty=False, padding_type='reflect',leaky_relu=False,seg= False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            if not leaky_relu:
                # print('*** relu ***')
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            elif leaky_relu:
                # print('*** leaky relu ***')
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.LeakyReLU(0.2, True)]


            # if uncertainty:
            #     model += [nn.Dropout(0.5)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, leaky_relu= leaky_relu)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            # if uncertainty:
            #     model += [nn.Dropout(0.5)]


        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        model += [nn.Tanh()]


        self.model = nn.Sequential(*model)

    def forward(self, input):
        # return self.model(input)
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,leaky_relu=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,leaky_relu=leaky_relu)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, leaky_relu):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if not leaky_relu:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]
        elif leaky_relu:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.LeakyReLU(0.2, True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# class ResnetGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',uncertainty=False,gpu_ids=[],leaky_relu=False,seg= False):
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
#                            bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         n_downsampling = 2
#         for i in range(n_downsampling):
#             mult = 2**i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
#                                 stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]
#
#         mult = 2**n_downsampling
#         for i in range(n_blocks):
#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
#
#         for i in range(n_downsampling):
#             mult = 2**(n_downsampling - i)
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1,
#                                          bias=use_bias),
#                       norm_layer(int(ngf * mult / 2)),
#                       nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         return self.model(input)
#
#
# # Define a resnet block
# class ResnetBlock(nn.Module):
#     def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,leaky_relu=False):
#         super(ResnetBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
#
#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         conv_block = []
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim),
#                        nn.ReLU(True)]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]
#
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim)]
#
#         return nn.Sequential(*conv_block)
#
#     def forward(self, x):
#         out = x + self.conv_block(x)
#         return out
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.n_layers = n_layers
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        else:
            sequence += [nn.Tanh()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

    def get_intermediate_features(self, input):
        intermediate_outputs = []
        conv_count = 0  # Keep track of how many convolutional layers we've passed through

        for i, submodule in enumerate(self.model.children()):
            if isinstance(submodule, nn.Conv2d):
                conv_count += 1
                input = submodule(input)

                # Skip the first and last conv layers
                if conv_count > 1 and conv_count < self.n_layers + 2:
                    intermediate_outputs.append(input)

        return intermediate_outputs




def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, mode='2d'):
        super(conv_block, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif mode == '3d':
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.conv = nn.Sequential(
            Conv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True),
            Conv(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, mode='2d'):
        super(up_conv, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif mode == '3d':
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2, mode='2d'):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        if mode == '2d':
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif mode == '3d':
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.conv = nn.Sequential(
            Conv(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2, mode='2d'):
        super(RRCNN_block, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
        elif mode == '3d':
            Conv = nn.Conv3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")

        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t, mode=mode),
            Recurrent_block(ch_out, t=t, mode=mode)
        )
        self.Conv_1x1 = Conv(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out, mode='2d'):
        super(single_conv, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif mode == '3d':
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")

        self.conv = nn.Sequential(
            Conv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, mode='2d'):
        super(Attention_block, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif mode == '3d':
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.W_g = nn.Sequential(
            Conv(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(F_int)
        )

        self.W_x = nn.Sequential(
            Conv(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(F_int)
        )

        self.psi = nn.Sequential(
            Conv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2,enable_dropout=False, mode='2d'):
        super(R2U_Net, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            MaxPool = nn.MaxPool2d
        elif mode == '3d':
            Conv = nn.Conv3d
            MaxPool = nn.MaxPool3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")

        self.enable_dropout = enable_dropout
        self.dropout = nn.Dropout(p=0.2)
        self.Maxpool = MaxPool(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t, mode=mode)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t, mode=mode)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t, mode=mode)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t, mode=mode)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t, mode=mode)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, mode=mode)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t, mode=mode)

        self.Up4 = up_conv(ch_in=512, ch_out=256, mode=mode)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t, mode=mode)

        self.Up3 = up_conv(ch_in=256, ch_out=128, mode=mode)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t, mode=mode)

        self.Up2 = up_conv(ch_in=128, ch_out=64, mode=mode)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, mode=mode)

        self.Conv_1x1 = Conv(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        if self.enable_dropout:
            d2 = self.dropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1,enable_dropout=False, mode='2d'):
        super(AttU_Net, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            MaxPool = nn.MaxPool2d
        elif mode == '3d':
            Conv = nn.Conv3d
            MaxPool = nn.MaxPool3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")

        self.enable_dropout = enable_dropout
        self.dropout = nn.Dropout(p=0.2)

        self.Maxpool = MaxPool(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64, mode=mode)
        self.Conv2 = conv_block(ch_in=64, ch_out=128, mode=mode)
        self.Conv3 = conv_block(ch_in=128, ch_out=256, mode=mode)
        self.Conv4 = conv_block(ch_in=256, ch_out=512, mode=mode)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024, mode=mode)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, mode=mode)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256, mode=mode)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, mode=mode)

        self.Up4 = up_conv(ch_in=512, ch_out=256, mode=mode)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128, mode=mode)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, mode=mode)

        self.Up3 = up_conv(ch_in=256, ch_out=128, mode=mode)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64, mode=mode)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, mode=mode)

        self.Up2 = up_conv(ch_in=128, ch_out=64, mode=mode)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32, mode=mode)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, mode=mode)

        self.Conv_1x1 = Conv(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        if self.enable_dropout:
            d2 = self.dropout(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2, enable_dropout=False, mode='2d'):
        print('ddd',enable_dropout)
        super(R2AttU_Net, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            MaxPool = nn.MaxPool2d
        elif mode == '3d':
            Conv = nn.Conv3d
            MaxPool = nn.MaxPool3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.enable_dropout = enable_dropout
        self.dropout = nn.Dropout(p=0.2)

        self.Maxpool = MaxPool(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t, mode=mode)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t, mode=mode)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t, mode=mode)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t, mode=mode)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t, mode=mode)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, mode=mode)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256, mode=mode)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t, mode=mode)

        self.Up4 = up_conv(ch_in=512, ch_out=256, mode=mode)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128, mode=mode)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t, mode=mode)

        self.Up3 = up_conv(ch_in=256, ch_out=128, mode=mode)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64, mode=mode)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t, mode=mode)

        self.Up2 = up_conv(ch_in=128, ch_out=64, mode=mode)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32, mode=mode)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, mode=mode)

        self.Conv_1x1 = Conv(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        # print('before R1', np.shape(x))
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        # print('before R2', np.shape(x2))

        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        # print('after R3', np.shape(x3))

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        # print('after R4', np.shape(x4))


        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        # print('after R5', np.shape(x5))


        # decoding + concat path
        d5 = self.Up5(x5)
        # print('after U5', np.shape(d5))

        x4 = self.Att5(g=d5, x=x4)
        # print('after U4', np.shape(x4))

        d5 = torch.cat((x4, d5), dim=1)
        # print('after cat5', np.shape(d5))

        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        if self.enable_dropout:
            d2 = self.dropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class conv_block_first(nn.Module):
    def __init__(self, ch_in, ch_out, mode='2d'):
        super(conv_block_first, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif mode == '3d':
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.conv = nn.Sequential(
            Conv(ch_in, ch_out, kernel_size=3, stride=1, padding='same', bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True),
            Conv(ch_out, ch_out, kernel_size=3, stride=1, padding='same', bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_second(nn.Module):
    def __init__(self, ch_in, ch_out, mode='2d'):
        super(conv_block_second, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
            Pool = nn.MaxPool2d
        elif mode == '3d':
            Conv = nn.Conv3d
            Pool = nn.MaxPool3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.conv = nn.Sequential(
            Conv(ch_in, ch_out, kernel_size=3, stride=2, padding=3, bias=True),
            Pool(3, stride=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv_first(nn.Module):
    def __init__(self, ch_in, ch_out, mode='2d'):
        super(up_conv_first, self).__init__()
        if mode == '2d':
            ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d
        elif mode == '3d':
            ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.up = nn.Sequential(

            ConvTranspose(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True, output_padding=1),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class up_conv_second(nn.Module):
    def __init__(self, ch_in, ch_out, mode='2d'):
        super(up_conv_second, self).__init__()
        if mode == '2d':
            ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d
        elif mode == '3d':
            ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.up = nn.Sequential(
            ConvTranspose(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True),
            ConvTranspose(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.up(x)
        return x


class up_conv_last(nn.Module):
    def __init__(self, ch_in, ch_out, mode='2d'):
        super(up_conv_last, self).__init__()
        if mode == '2d':
            Conv = nn.Conv2d
        elif mode == '3d':
            Conv = nn.Conv3d
        else:
            raise ValueError("Mode must be '2d' or '3d'")
        self.up = nn.Sequential(
            Conv(48, ch_out, kernel_size=1, stride=1, padding=0),
            nn.Softmax()

        )

    def forward(self, x):
        x = self.up(x)
        return x



class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2,enable_dropout=False, mode='2d'):
        super(U_Net, self).__init__()

        # self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.enable_dropout = enable_dropout
        self.dropout = nn.Dropout(p=0.2)

        self.Conv1 = conv_block_first(ch_in=img_ch, ch_out=48, mode=mode)
        self.Conv2 = conv_block_second(ch_in=48, ch_out=48, mode=mode)

        self.Conv3 = conv_block_first(ch_in=48, ch_out=96, mode=mode)
        self.Conv4 = conv_block_second(ch_in=96, ch_out=96, mode=mode)

        self.Conv5 = conv_block_first(ch_in=96, ch_out=192, mode=mode)
        self.Conv6 = conv_block_second(ch_in=192, ch_out=192, mode=mode)

        self.Conv7 = conv_block_first(ch_in=192, ch_out=384, mode=mode)

        self.Up_8 = up_conv_first(ch_in=384, ch_out=192, mode=mode)
        self.Up_conv9 = up_conv_second(ch_in=384, ch_out=192, mode=mode)

        self.Up_10 = up_conv_first(ch_in=192, ch_out=96, mode=mode)
        self.Up_conv11 = up_conv_second(ch_in=192, ch_out=96, mode=mode)

        self.Up_12 = up_conv_first(ch_in=96, ch_out=48, mode=mode)
        self.Up_conv13 = up_conv_second(ch_in=96, ch_out=48, mode=mode)

        self.Up_14 = up_conv_last(ch_in=48, ch_out=output_ch, mode=mode)

    def forward(self, x):
        # encoding path
        # print('before Conv1', np.shape(x))
        x1 = self.Conv1(x)
        # print('Conv1',np.shape(x1))

        x2 = self.Conv2(x1)
        # print('Conv2',np.shape(x2))

        x3 = self.Conv3(x2)
        # print('Conv3',np.shape(x3))

        x4 = self.Conv4(x3)
        # print('Conv4',np.shape(x4))

        x5 = self.Conv5(x4)
        # print('Conv5',np.shape(x5))

        x6 = self.Conv6(x5)

        # print('Conv6',np.shape(x6))

        x7 = self.Conv7(x6)
        # print('Conv7',np.shape(x7))

        # decoding + concat path
        d8 = self.Up_8(x7)
        # print('Convd8',np.shape(d8))

        d8 = torch.cat((x5, d8), dim=1)
        # print('catd8,x5',np.shape(d8))

        d9 = self.Up_conv9(d8)
        # print('catd9',np.shape(d9))

        d10 = self.Up_10(d9)
        # print('catd10',np.shape(d10))

        d10 = torch.cat((x3, d10), dim=1)
        # print('catd10,3',np.shape(d10))

        d11 = self.Up_conv11(d10)
        # print('catd11',np.shape(d11))

        d12 = self.Up_12(d11)
        # print('catd12',np.shape(d12))

        d12 = torch.cat((x1, d12), dim=1)
        # print('catd12',np.shape(d12))

        d13 = self.Up_conv13(d12)
        # print('catd13',np.shape(d13))
        if self.enable_dropout:
            d13 = self.dropout(d13)

        d14 = self.Up_14(d13)
        # print('catd14',np.shape(d14))

        return d14

class GANLoss_papar(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(
        self, gan_mode = 'lsgan', target_real_label = 1.0, target_fake_label = 0.0
    ):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) -- the type of GAN objective.
                Choices: vanilla, lsgan, and wgangp.
            target_real_label (bool) -- label for a real image
            target_fake_label (bool) -- label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. Vanilla GANs will handle it with
        BCEWithLogitsLoss.
        """
        super().__init__()

        # pylint: disable=not-callable
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) -- tpyically the prediction from a
                discriminator
            target_is_real (bool) -- if the ground truth label is for real
                images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of
            the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) -- tpyically the prediction output from a
                discriminator
            target_is_real (bool) -- if the ground truth label is for real
                images or fake images

        Returns:
            the calculated loss.
        """

        if self.gan_mode == 'wgan':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

# pylint: disable=too-many-arguments
# pylint: disable=redefined-builtin
def cal_gradient_penalty(
    netD, real_data, fake_data, device,
    type = 'mixed', constant = 0, lambda_gp = 0.1
):
    """Calculate the gradient penalty loss, used in WGAN-GP

    source: https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- torch device
        type (str)                  -- if we mix real and fake data or not
            Choices: [real | fake | mixed].
        constant (float)            -- the constant used in formula:
            (||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp == 0.0:
        return 0.0, None

    lambda_gp = lambda_gp #/ constant**2
    if type == 'real':
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1, device = device)
        alpha = alpha.expand(
            real_data.shape[0], real_data.nelement() // real_data.shape[0]
        ).contiguous().view(*real_data.shape)

        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))

    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolatesv,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )

    gradients = gradients[0].view(real_data.size(0), -1)

    gradient_penalty = (
        ((gradients + 1e-16).norm(2, dim=1) - constant) ** 2
    ).mean() * lambda_gp

    return gradient_penalty, gradients
