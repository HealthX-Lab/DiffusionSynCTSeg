import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

from monai.networks import normal_init
from monai.networks.nets import SwinUNETR
from monai.networks.nets import UNet
from monai.networks.nets import SegResNet, SegResNetVAE
from monai.networks.nets import Discriminator
###############################################################################
# Functions
###############################################################################


def weights_init(net, init_type='normal', gain=0.02):
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
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

# class CustomSegResNet(SegResNet):
#     def __init__(self,attributes):
#         super().__init__(**attributes)
#         out_channel = attributes["out_channels"]
#         self.dropout = nn.Dropout(p=opt.Gen_dropout)
#         self.final_conv = nn.Conv3d(out_channel, out_channel, 1, bias=False)
#         self.final_act = nn.Softmax()
#
#     def forward(self, x):
#         x = super(SegResNet, self).forward(x)
#         x = self.final_conv(x)
#         x = self.dropout(x)
#         x = self.final_act(x)
#         return x

# class CustomSegResNetVAE(SegResNetVAE):
#     def __init__(self,attributes):
#         super().__init__(**attributes)
#         out_channel = attributes["out_channels"]
#         self.dropout = nn.Dropout(p=opt.Gen_dropout)
#         self.final_conv = nn.Conv3d(out_channel, out_channel, 1 ,bias=False)
#         self.final_act = nn.Softmax()
#
#     def forward(self, x):
#         x = super(SegResNetVAE, self).forward(x)
#         x = self.final_conv(x)
#         x = self.dropout(x)
#         x = self.final_act(x)
#         return x

# class CustomUNet(UNet):
#     def __init__(self,attributes):
#         super().__init__(**attributes)
#         out_channel = attributes["out_channels"]
#         self.dropout = nn.Dropout(p=opt.Gen_dropout)
#         self.final_conv = nn.Conv3d(out_channel, out_channel, 1 ,bias=False)
#         self.final_act = nn.Softmax()
#
#     def forward(self, x):
#         x = super(UNet, self).forward(x)
#         x = self.final_conv(x)
#         x = self.dropout(x)
#         x = self.final_act(x)
#         return x

# def define_G(opt,seg_net=False):
#     netG = None
#     which_model_netG = opt.which_model_netG if not seg_net else which_model_netG_seg
#
#     if which_model_netG == 'SegResNetVAE':
#         if not seg_net:
#             netG = CustomSegResNetVAE(**opt.SegResNetVAE_meatdata)
#         else:
#             netG = CustomSegResNetVAE(**opt.SegResNetVAE_SEG_meatdata)
#     elif which_model_netG == 'SegResNet':
#         if not seg_net:
#             netG = CustomSegResNet(**opt.SegResNet_meatdata)
#         else:
#             netG = CustomSegResNet(**opt.SegResNet_SEG_meatdata)
#     elif which_model_netG == 'UNet':
#         if not seg_net:
#             netG = CustomUNet(**opt.UNet_meatdata)
#         else:
#             netG = CustomUNet(**opt.UNet_SEG_meatdata)
#     elif which_model_netG == 'SwinUNETR':
#         if not seg_net:
#             netG = CustomSwinUNETR(**opt.SwinUNETR_meatdata)
#         else:
#             netG = CustomSwinUNETR(**opt.SwinUNETR_SEG_meatdata)
#     else:
#         raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
#
#     netG = netG.to(opt.device)
#     netG.apply(normal_init)
#     return netG


def define_G(opt,seg_net=False):
    netG = None
    which_model_netG = opt.which_model_netG if not seg_net else opt.which_model_netG_seg

    if which_model_netG == 'SegResNetVAE':
        if not seg_net:
            netG = SegResNetVAE(**opt.SegResNetVAE_meatdata)
        else:
            netG = SegResNetVAE(**opt.SegResNetVAE_SEG_meatdata)
    elif which_model_netG == 'SegResNet':
        if not seg_net:
            netG = SegResNet(**opt.SegResNet_meatdata)
        else:
            netG = SegResNet(**opt.SegResNet_SEG_meatdata)
    elif which_model_netG == 'UNet':
        if not seg_net:
            netG = UNet(**opt.UNet_meatdata)
        else:
            netG = UNet(**opt.UNet_SEG_meatdata)
    elif which_model_netG == 'SwinUNETR':
        if not seg_net:
            netG = SwinUNETR(**opt.SwinUNETR_meatdata)
        else:
            netG = SwinUNETR(**opt.SwinUNETR_SEG_meatdata)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    netG = netG.to(opt.device)
    netG.apply(normal_init)
    return netG


class CustomDiscriminator(Discriminator):
    def __init__(self,opt):
        super().__init__(**opt.Discriminator_basic_metadata)
        self.final_conv = nn.Conv1d(1, 1, 1, bias=False)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        print('****final D input*****', np.shape(x))
        x = super(Discriminator, self).forward(x)
        print('****final D outout*****',np.shape(x))
        # x = Discriminator.forward(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        print('****final D outout act*****', np.shape(x))
        return x

def define_D(opt):
    netD = None
    which_model_netD = opt.which_model_netD
    if which_model_netD == 'basic':
        netD = CustomDiscriminator(opt)
    netD = netD.to(opt.device)
    netD.apply(normal_init)
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
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = torch.ones_like(input)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = torch.zeros_like(input)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        print('**** GAN LOSS input', np.shape(input))
        target_tensor = self.get_target_tensor(input, target_is_real)
        print('**** GAN LOSS target_tensor', np.shape(target_tensor))
        print('**** GAN LOSS target_tensor val ', self.loss(input, target_tensor),np.shape(self.loss(input, target_tensor)))
        return self.loss(input, target_tensor)

