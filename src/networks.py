import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


from monai.networks.nets import SwinUNETR
from monai.networks.nets import UNet
from monai.networks.nets import SegResNet, SegResNetVAE
from monai.networks.nets import Discriminator
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# def get_norm_layer(norm_type='instance'):
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm)
#     return norm_layer


def define_G(opt,seg_net=False):
    netG = None
    which_model_netG = opt.which_model_netG

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = netG.to(device)
    netG.apply(weights_init)
    return netG


def define_D(opt):
    netD = None
    which_model_netD = opt.which_model_netD
    if which_model_netD == 'basic':
        netD = Discriminator(**opt.Discriminator_basic_meatdata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netD = netD.to(device)
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
            self.loss = nn.MSELoss()
        else:
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

