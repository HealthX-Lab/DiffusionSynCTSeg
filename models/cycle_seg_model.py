import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import torch.nn as nn
import sys
import skimage
import numpy as np
# from . import ssim
from pytorch_msssim import ssim


def compute_perceptual_loss(fake_images, real_images, discriminator):
    # Extract features from discriminator layers
    fake_features = discriminator.get_intermediate_features(fake_images)
    real_features = discriminator.get_intermediate_features(real_images)

    # Initialize perceptual loss
    p_loss = 0

    # Calculate perceptual loss as the MSE loss between the intermediate features of fake and real images
    for f, r in zip(fake_features, real_features):
        p_loss += F.mse_loss(f, r)

    return p_loss


def weighted_dice_loss(probs1, probs2, weights=[0.2,0.8]):
    """
    probs1, probs2 are torch variables of size BatchxnclassesxHxW representing probabilities for each class
    weights is a tensor representing the weight for each class
    """
    # print('min and mx *** ',probs2.min(),probs2.max())
    threshold = 0
    probs2 = torch.where(probs2 > threshold, torch.tensor(1.0).cuda(), torch.tensor(-1.0).cuda())
    # print('min and mx +++ ', probs2.min(),probs2.max())

    weights = torch.from_numpy(np.array(weights)).cuda().float()
    # Ensure that the input tensors have the same size
    assert probs1.size() == probs2.size(), "Input sizes must be equal."
    # Ensure that the input tensors are 4D
    assert probs1.dim() == 4, "Input must be a 4D Tensor."

    # Calculate the overlap between the predicted probabilities
    num = probs1 * probs2  # b,c,h,w--p1*p2
    num = torch.sum(num, dim=3)  # Sum over the height dimension
    num = torch.sum(num, dim=2)  # Sum over the width dimension
    num = torch.sum(num, dim=0)  # Sum over the batch dimension

    # If weights are provided, apply them to the numerator of the Dice coefficient calculation
    if weights is not None:
        num = weights * num

    # Calculate the total 'area' of the predicted probabilities for probs1
    den1 = probs1 * probs1  # --p1^2
    den1 = torch.sum(den1, dim=3)  # Sum over the height dimension
    den1 = torch.sum(den1, dim=2)  # Sum over the width dimension
    den1 = torch.sum(den1, dim=0)  # Sum over the batch dimension

    # Calculate the total 'area' of the predicted probabilities for probs2
    den2 = probs2 * probs2  # --p2^2
    den2 = torch.sum(den2, dim=3)  # Sum over the height dimension
    den2 = torch.sum(den2, dim=2)  # Sum over the width dimension
    den2 = torch.sum(den2, dim=0)  # Sum over the batch dimension

    # Calculate the Dice coefficient for each class
    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))  # Add epsilon to avoid division by zero

    # Exclude the Dice coefficient for the background class
    dice_eso = dice[0:]



    # Calculate the Dice loss by taking the negative of the average of the Dice coefficients for the foreground classes
    dice_total = 1 -  torch.sum(dice_eso) / dice_eso.size(0)  # divide by number of classes (batch_sz)

    return dice_total

def CrossEntropyLoss2d(inputs, targets, weight=None, size_average=True):
    lossval = 0
    nll_loss = nn.NLLLoss2d(weight, size_average)
    for output, label in zip(inputs, targets):
        lossval += nll_loss(F.log_softmax(output), label)
    return lossval

def CrossEntropy2d(input, target, weight=None, size_average=True):
    # print('** in cross entropy',input.shape,target.shape)
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    # loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    # loss = F.cross_entropy(input, target, weight=weight)
    # print('target_mask.sum().data[0]',target_mask.sum().item())
    if size_average:
        loss /= target_mask.sum().item()
        # print('size of cross entropy',loss, target_mask.sum().item())

    return loss
#
def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def dice_loss_norm(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # print('** in dice',input.shape,target.shape)

    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input,dim=1)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  #
    num = torch.sum(num, dim=0)# b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[1:]  # we ignore bg dice val, and take the fg
    dice_total = 1 -  torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    return dice_total


class CycleSEGModel(BaseModel):
    def name(self):
        return 'CycleSEGModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        opt.no_dropout = False

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_Seg = self.Tensor(nb, opt.output_nc_seg, size, size)

        if opt.seg_norm == 'CrossEntropy' or self.opt.seg_norm =='CombinationLoss':
            self.input_Seg_one = self.Tensor(nb, opt.output_nc, size, size)


        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,uncertainty=opt.uncertainty,leaky_relu = opt.leaky_relu,seg=False)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,uncertainty=opt.uncertainty,leaky_relu = opt.leaky_relu,seg=False)

        self.netG_seg = networks.define_G(opt.input_nc_seg, opt.output_nc_seg,
                                        opt.ngf, opt.which_model_netSeg, opt.norm, not opt.no_dropout, self.gpu_ids,uncertainty=opt.uncertainty,leaky_relu = opt.leaky_relu,seg=True,mode=opt.mode)

        if opt.seg_rec_loss or opt.seg_fakeMRI_realCT_loss:
            self.netG_seg_mri = networks.define_G(opt.input_nc_seg, opt.output_nc_seg,
                                              opt.ngf, opt.which_model_netSeg, opt.norm, not opt.no_dropout,
                                              self.gpu_ids, uncertainty=opt.uncertainty, leaky_relu=opt.leaky_relu,
                                              seg=True)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        print('ressss',not self.isTrain , opt.continue_train)
        if not self.isTrain or opt.continue_train:
            print('helloooooo')

            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netG_seg, 'Seg_A', which_epoch)



                    # just_train_seg = True
        if opt.just_segmentation:
            for param in self.netG_A.parameters():
                param.requires_grad = False
            for param in self.netG_B.parameters():
                param.requires_grad = False
            for param in self.netD_A.parameters():
                param.requires_grad = False
            for param in self.netD_B.parameters():
                param.requires_grad = False

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.gradient_penalty = opt.gradient_penalty
            if opt.Wasserstein_Lossy:
                self.criterionGAN = networks.WassersteinLoss()
            elif opt.gradient_penalty:
                self.criterionGAN = networks.GANLoss_papar().cuda()#.to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, target_real_label=1.0, target_fake_label=0.0)

            if opt.mode =='3d':
                arr = np.array(opt.crossentropy_weight)
                weight = torch.from_numpy(arr).cuda().float()
                self.criterion_seg = nn.CrossEntropyLoss(weight=weight)
                Depth = opt.Depth
                nb = opt.batchSize
                size = opt.fineSize
                self.input_Seg_one = self.Tensor(size=(nb, 1, Depth, size, size))
                self.fake_B_3d = self.Tensor(nb, 1, Depth, size, size)
                self.input_B = self.Tensor(nb, 1, size, size)

            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            if opt.MIND_loss:
                self.criterionMIND = networks.MINDLoss(non_local_region_size=opt.non_local_region_size,
                                                       patch_size=opt.patch_size, neighbor_size=opt.neighbor_size,
                                                       gaussian_patch_sigma=opt.gaussian_patch_sigma,loss_type=opt.MIND_loss_type).cuda()

            if opt.boundry_loss:
                self.surface_loss = networks.SurfaceLoss(idc=[1])

            # initialize optimizers
            if not self.opt.separate_segmentation:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_seg.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)#,amsgrad=True)

                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)#,amsgrad=True)
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)#,amsgrad=True)

            elif self.opt.separate_segmentation:
                self.optimizer_G = torch.optim.Adam(
                    itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_seg = torch.optim.Adam(self.netG_seg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))





            if self.opt.segmentation_discriminator:
                self.netD_seg = networks.define_D(opt.input_nc, opt.ndf,
                                  opt.which_model_netD,
                                  opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
                self.criterionSEG = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, target_real_label=1.0, target_fake_label=0.0)
                self.optimizer_D_seg = torch.optim.Adam(self.netD_seg.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))



        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_seg)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')




        # early stopping
        if self.opt.enable_early_stopping:
            print('in early stopping  &&&&')
            self.best_metric = {'G_A': float('inf'), 'G_B': float('inf'), 'D_A': float('inf'), 'D_B': float('inf'), 'seg': float('inf')}
            self.epochs_since_improvement = {'G_A': 0, 'G_B': 0, 'D_A': 0, 'D_B': 0, 'seg': 0}
            self.disable_training = {'G_A': 0, 'G_B': 0, 'D_A': 0, 'D_B': 0, 'seg': 0}
            self.stop_training = 0
        if self.opt.eval_step:
            self.best_mse_fake_A = self.best_mse_fake_B = self.best_mse_rec_A = self.best_mse_rec_B = float('inf')
            self.best_MI_fake_A = self.best_MI_fake_B = self.best_ssim_fake_A = self.best_ssim_fake_B =\
                self.best_ssim_rec_A = self.best_ssim_rec_B = self.best_dice_seg_real = 0



    def get_enable_training_model(self):
        list_enable = []
        for key, value in self.disable_training.items():
            if value == 0:
                list_enable.append(key)
        return list_enable


    def set_input(self, input,d=0):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        input_B = input['B' if AtoB else 'A']
        input_Seg = input['Seg']
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.dist_map_label = input['dist_map']

        if self.opt.mode =='2d':
            # print('input_Seg', input_Seg.shape, '*****')

            self.input_Seg.resize_(input_Seg.size()).copy_(input_Seg)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
            # print(self.image_paths)
            if (self.opt.seg_norm == 'CrossEntropy' or self.opt.seg_norm =='CombinationLoss') :
                input_Seg_one = input['Seg_one']
                self.input_Seg_one.resize_(input_Seg_one.size()).copy_(input_Seg_one)

        elif self.opt.mode =='3d':
            self.data_number = d
            seg_2D = input['Seg_one']

            if d%41 < self.opt.Depth:
                # print('in setting 3D seg in input ', d%41)
                self.input_Seg_one[:, :, d%41, :, :] = seg_2D
                # self.real_B_3d[:, :, d%41, :, :] = input_B



    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_Seg = Variable(self.input_Seg)
        if (self.opt.seg_norm == 'CrossEntropy' or self.opt.seg_norm =='CombinationLoss') :
            self.real_Seg_one = Variable(self.input_Seg_one.long())
            # print('self.real_Seg_one',self.real_Seg_one.shape, '***&*')


    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_D_basic(self, netD, real, fake):
        loss_D = 0
        if not self.opt.Wasserstein_Lossy:
            # Real
            pred_real = netD.forward(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake = netD.forward(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        if self.gradient_penalty is not None:
            gradient_p = networks.cal_gradient_penalty(
                netD, real, fake, real.device
            )[0]
            # print('***&*&',gradient_p)
            loss_D += gradient_p

        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        # fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * self.opt.lambda_D

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        # fake_A = self.fake_A
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_D

    def backward_G(self,data_number=0,epoch=0):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(self.fake_B)

        if not self.opt.Wasserstein_Lossy:
            self.loss_G_A = self.criterionGAN(pred_fake, True) * self.opt.lambda_D

        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)

        if not self.opt.Wasserstein_Lossy:
            self.loss_G_B = self.criterionGAN(pred_fake, True)  * self.opt.lambda_D

        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        if self.opt.MIND_loss:
            self.loss_mind_A = 0
            self.loss_mind_B = 0
            if self.opt.MIND_sameModalityLoss:
                self.loss_mind_A += self.criterionMIND(self.fake_A, self.real_A) * lambda_B * self.opt.lambda_mind * self.opt.MIND_sameModalityLossWeight
                self.loss_mind_B += self.criterionMIND(self.fake_B, self.real_B) * lambda_A * self.opt.lambda_mind * self.opt.MIND_sameModalityLossWeight
            if self.opt.MIND_diffModalityLoss:
                self.loss_mind_A += self.criterionMIND(self.fake_B, self.real_A) * lambda_B * self.opt.lambda_mind * self.opt.MIND_diffModalityLossWeight
                self.loss_mind_B += self.criterionMIND(self.fake_A, self.real_B) * lambda_A * self.opt.lambda_mind * self.opt.MIND_diffModalityLossWeight
        else:
            self.loss_mind_A = 0
            self.loss_mind_B = 0

        if self.opt.lambda_cc > 0:
            self.loss_CC_A = 0
            self.loss_CC_B = 0
            if self.opt.CC_sameModalityLoss:
                self.loss_CC_A += networks.Cor_CoeLoss(self.fake_A, self.real_A) * lambda_B * self.opt.lambda_cc * self.opt.MIND_sameModalityLossWeight
                self.loss_CC_B += networks.Cor_CoeLoss(self.fake_B, self.real_B) * lambda_A * self.opt.lambda_cc * self.opt.MIND_sameModalityLossWeight
            if self.opt.CC_diffModalityLoss:
                self.loss_CC_A += networks.Cor_CoeLoss(self.fake_B, self.real_A) * lambda_B * self.opt.lambda_cc * self.opt.MIND_diffModalityLossWeight
                self.loss_CC_B += networks.Cor_CoeLoss(self.fake_A, self.real_B) * lambda_A * self.opt.lambda_cc * self.opt.MIND_diffModalityLossWeight

        else:
                self.loss_CC_A = 0
                self.loss_CC_B = 0

        # Segmentation loss
        if self.opt.which_model_netSeg =='swin':
            print('***',self.fake_B.shape, self.real_A.shape,  self.real_Seg_one.shape)
            self.fake_B = F.interpolate(self.fake_B, size=(224, 224), mode='bilinear', align_corners=False)
            self.real_A = F.interpolate(self.real_A, size=(224, 224), mode='bilinear', align_corners=False)
            self.real_Seg_one = F.interpolate(self.real_Seg_one.float(), size=(224, 224), mode='nearest').long()
            print('###',self.fake_B.shape, self.real_A.shape,  self.real_Seg_one.shape)

        if not self.opt.separate_segmentation:
            if self.opt.mode=='2d':
                fake_B = self.fake_B
                if self.opt.add_quantum_noise:
                    fake_B = self.add_poisson_noise(fake_B)
            self.seg_fake_B = self.netG_seg.forward(fake_B)
        else:
            if self.opt.mode=='2d':
                fake_B = self.fake_B.detach()
                if self.opt.add_quantum_noise:
                    fake_B = self.add_poisson_noise(fake_B)
                self.seg_fake_B = self.netG_seg.forward(fake_B)
            elif self.opt.mode=='3d':
                d = self.data_number
                fake_B = self.fake_B.detach()
                if d % 41 < self.opt.Depth:
                    # print(' in setting 3d fake B')
                    self.fake_B_3d[:, :, d % 41, :, :] = fake_B
                if self.data_number % 40 == 0 and self.data_number:
                    # print(' in segmentaion 3d')
                    self.fake_seg = self.netG_seg(self.fake_B_3d)


        if self.opt.seg_norm == 'DiceNorm':
            self.loss_seg = dice_loss_norm(self.seg_fake_B, self.real_Seg)
            self.loss_seg = self.loss_seg * self.opt.weight_segmentation_in_GAN
        elif self.opt.seg_norm == 'CrossEntropy':
            if self.opt.mode == '2d':
                arr = np.array(self.opt.crossentropy_weight)
                weight = torch.from_numpy(arr).cuda().float()
                self.loss_seg = CrossEntropy2d(self.seg_fake_B, self.real_Seg_one, weight=weight) * self.opt.weight_segmentation_in_GAN
                if self.opt.MRI_CT_segmentation:
                    self.seg_real_A = self.netG_seg.forward(self.real_A)

                    self.loss_seg += CrossEntropy2d(self.seg_real_A, self.real_Seg_one,
                                                   weight=weight) * self.opt.weight_segmentation_in_GAN
                    self.loss_seg = self.loss_seg/2.0
            elif self.opt.mode == '3d':
                if self.data_number % 40 == 0 and self.data_number:
                    print(self.fake_seg.shape, self.real_Seg.shape)

                    self.loss_seg = self.criterion_seg(self.fake_seg, self.real_Seg_one)




        elif self.opt.seg_norm =='CombinationLoss':
            self.loss_seg = dice_loss_norm(self.seg_fake_B, self.real_Seg)
            dice_loss = self.loss_seg
            arr = np.array(self.opt.crossentropy_weight)
            weight = torch.from_numpy(arr).cuda().float()
            CE_loss = CrossEntropy2d(self.seg_fake_B, self.real_Seg_one, weight=weight)

            self.loss_seg = 1.0 * CE_loss + 0.2 * dice_loss

        if self.opt.boundry_loss:
            self.boundry_seg_loss = self.surface_loss(F.softmax(self.seg_fake_B, dim=1),self.dist_map_label.cuda())/20.0
            b_weight = self.opt.boundry_loss_weight
            self.loss_seg = self.loss_seg +  self.boundry_seg_loss*b_weight


        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A +
                       self.loss_idt_B  + self.loss_mind_A + self.loss_mind_B + self.loss_CC_A +
                       self.loss_CC_B)
        if not self.opt.separate_segmentation:

            self.loss_G +=  self.loss_seg
        if self.opt.which_model_netSeg =='swin':
            print('***',self.fake_B.shape, self.real_A.shape,  self.real_Seg_one.shape)
            self.fake_B = F.interpolate(self.fake_B, size=(256, 256), mode='bilinear', align_corners=False)
            self.real_A = F.interpolate(self.real_A, size=(256, 256), mode='bilinear', align_corners=False)


        self.loss_G.backward()






    def optimize_parameters(self,data_number,epoch=0):
        if not self.opt.enable_early_stopping:
            # forward
            self.forward()
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)
            self.optimizer_G.zero_grad()
            if self.opt.separate_segmentation :
                if self.opt.mode=='2d'or (self.opt.mode=='3d'and self.data_number%40 ==0):
                    self.optimizer_seg.zero_grad()

            self.backward_G(data_number)
            self.optimizer_G.step()
            if self.opt.separate_segmentation :
                if self.opt.mode=='2d'or (self.opt.mode=='3d'and self.data_number%40 ==0 and self.data_number):
                    self.loss_seg.backward()
                    self.optimizer_seg.step()

            # D_A
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            self.optimizer_D_A.step()
            # D_B
            self.optimizer_D_B.zero_grad()
            self.backward_D_B()
            self.optimizer_D_B.step()

        elif self.opt.enable_early_stopping:
            if len(self.get_enable_training_model()) == 0:
                self.stop_training = 1
                return
            self.forward()
            # G_A and G_B
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            if not self.disable_training['D_A']:
                self.optimizer_D_A.zero_grad()
                self.backward_D_A()
                self.optimizer_D_A.step()
            if not self.disable_training['D_B']:
                self.optimizer_D_B.zero_grad()
                self.backward_D_B()
                self.optimizer_D_B.step()

    def get_current_errors(self):
        D_A = self.loss_D_A.item()
        G_A = self.loss_G_A.item()
        Cyc_A = self.loss_cycle_A.item()
        D_B = self.loss_D_B.item()
        G_B = self.loss_G_B.item()
        Cyc_B = self.loss_cycle_B.item()
        Seg_B = self.loss_seg.item()
        error_list = [ ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('G_B', G_B), ('Cyc_B', Cyc_B),
                                ('Seg', Seg_B),('D_B', D_B),('D_A', D_A)]# ('D_B', D_B),('D_A', D_A),
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.item()
            idt_B = self.loss_idt_B.item()
            error_list.append(('idt_B', idt_B))
            error_list.append(('idt_A', idt_A))

        if self.opt.seg_rec_loss:
            loss_seg_rec_mri = self.loss_seg_mri.item()
            error_list.append(('seg_rec_mri', loss_seg_rec_mri))
        if self.opt.seg_fakeMRI_realCT_loss:
            error_list.append(('loss_seg_fake_mri_real_CT', self.loss_seg_fake_mri_real_CT.item()))
            error_list.append(('loss_seg_rec_mri', self.loss_seg_rec_mri.item()))
            error_list.append(('loss_seg_real_mri', self.loss_seg_real_mri.item()))
        if self.opt.perceptual_loss:
            error_list.append(('perceptual_loss', self.p_loss.item()))
        if self.opt.direct_loss:
            error_list.append(('loss_cycled_A', self.loss_cycled_A.item()))
            error_list.append(('loss_cycled_B', self.loss_cycled_B.item()))

        if self.opt.MIND_loss:
            error_list.append(('mind_A', self.loss_mind_A.item()))
            error_list.append(('mind_B', self.loss_mind_B.item()))

        if self.opt.lambda_cc>0:
            error_list.append(('cc_A', self.loss_CC_A.item()))
            error_list.append(('cc_B', self.loss_CC_B.item()))

        return OrderedDict(error_list)

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        seg_B = util.tensor2seg(torch.max(self.seg_fake_B.data,dim=1,keepdim=True)[1])
        manual_B = util.tensor2seg(torch.max(self.real_Seg.data,dim=1,keepdim=True)[1])
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        visual_list = [('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),('seg_B',seg_B), ('manual_B',manual_B)]
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            visual_list.append(('idt_A', idt_A))
            visual_list.append(('idt_B', idt_B))

        if self.opt.seg_rec_loss:
            seg_rec_mri = util.tensor2seg(torch.max(self.seg_rec_A.data, dim=1, keepdim=True)[1])
            visual_list.append(('seg_rec_mri', seg_rec_mri))



        return visual_list

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netG_seg, 'Seg_A', label, self.gpu_ids)
        if self.opt.seg_rec_loss:
            self.save_network(self.netG_seg_mri, 'Seg_A_mri', label, self.gpu_ids)

    def save_individual_model(self, model_name, label):
        # print(' in the save model ', label )
        if model_name == 'G_A':
            self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        elif model_name == 'D_A':
            self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        elif model_name == 'G_B':
            self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        elif model_name == 'D_B':
            self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        elif model_name == 'Seg_A':
            self.save_network(self.netG_seg, 'Seg_A', label, self.gpu_ids)

    def earlyStopping(self, epoch):
        loss_G_A = 0
        loss_G_B = 0
        if self.opt.identity > 0.0:
            loss_G_A = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A
            loss_G_B = self.loss_G_B + self.loss_cycle_B + self.loss_idt_B
        else:
            loss_G_A = self.loss_G_A + self.loss_cycle_A
            loss_G_B = self.loss_G_B + self.loss_cycle_B

        print('%^%^ in early stopping')
        if not self.disable_training['G_A']:
            self.check_eraly_stopping('G_A', self.netG_A, loss_G_A, self.opt.min_delta_G, epoch)
        if not self.disable_training['G_B']:
            self.check_eraly_stopping('G_B', self.netG_B, loss_G_B, self.opt.min_delta_G, epoch)

        if not self.disable_training['D_A']:
            self.check_eraly_stopping('D_A', self.netD_A, self.loss_D_A, self.opt.min_delta_D, epoch)
        if not self.disable_training['D_B']:
            self.check_eraly_stopping('D_B', self.netD_B, self.loss_D_B, self.opt.min_delta_D, epoch)

        if not self.disable_training['seg']:
            self.check_eraly_stopping('Seg_A', self.netG_seg, self.loss_seg, self.opt.min_delta_seg, epoch)

    def check_eraly_stopping(self,name,model,loss,delta,epoch):
        print('in check early stopping ', name)
        if self.best_metric[name] - loss >=  delta:
            self.best_metric[name] = loss
            self.epochs_since_improvement[name] = 0
            # Save the best model
            self.save_individual_model(name,f'best_{name}')
        else:
            print('I am in plus for ', name)
            self.epochs_since_improvement[name] += 1

        patience = 0
        if 'D' in name:
            if self.epochs_since_improvement[name] >= self.opt.patience_D and epoch<60:
                self.opt.patience_D = self.opt.patience_D + 5

            patience = self.opt.patience_D
        elif 'G_':
            if self.epochs_since_improvement[name] >= self.opt.patience_G and epoch<60:
                self.opt.patience_G = self.opt.patience_G + 5

            patience = self.opt.patience_G
        elif 'seg':
            if self.epochs_since_improvement[name] >= self.opt.patience_seg and epoch<60:
                self.opt.patience_seg = self.opt.patience_seg + 5

            patience = self.opt.patience_seg



        if self.epochs_since_improvement[name] >= patience:
            print(f"Early stopping for {name}in epoch {epoch} patience {patience}!")
            self.save_individual_model(name, f'best_{name}')
            for param in model.parameters():
                param.requires_grad = False
            self.disable_training[name] = 1

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr




    def test(self,input):
        with torch.no_grad():
            size = self.opt.fineSize
            v_batch = self.opt.eval_batch
            self.input_A_val = self.Tensor(v_batch, self.opt.input_nc, size, size)
            self.input_B_val = self.Tensor(v_batch, self.opt.output_nc, size, size)
            self.input_Seg_val = self.Tensor(v_batch, self.opt.output_nc_seg, size, size)
            AtoB = self.opt.which_direction == 'AtoB'
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
            input_Seg = input['Seg']
            self.input_A_val.resize_(input_A.size()).copy_(input_A)
            self.input_B_val.resize_(input_B.size()).copy_(input_B)
            self.input_Seg_val.resize_(input_Seg.size()).copy_(input_Seg)



            self.real_A_val = Variable(self.input_A_val)
            self.fake_B_val = self.netG_A.forward(self.real_A_val)
            self.rec_A_val = self.netG_B.forward(self.fake_B_val)

            self.real_B_val = Variable(self.input_B_val)
            self.fake_A_val = self.netG_B.forward(self.real_B_val)
            self.rec_B_val = self.netG_A.forward(self.fake_A_val)

            if self.opt.which_model_netSeg == 'swin':
                print('***', self.fake_B.shape, self.real_A.shape, self.real_Seg_one.shape)
                self.fake_B_val = F.interpolate(self.fake_B_val, size=(224, 224), mode='bilinear', align_corners=False)
                self.real_B_val = F.interpolate(self.real_B_val, size=(224, 224), mode='bilinear', align_corners=False)
            self.seg_fake_B_val = self.netG_seg.forward(self.fake_B_val)
            self.seg_real_B_val = self.netG_seg.forward(self.real_B_val)
            if self.opt.which_model_netSeg == 'swin':
                print('***', self.fake_B.shape, self.real_A.shape, self.real_Seg_one.shape)
                self.fake_B_val = F.interpolate(self.fake_B_val, size=(256, 256), mode='bilinear', align_corners=False)
                self.real_B_val = F.interpolate(self.real_B_val, size=(256, 256), mode='bilinear', align_corners=False)






    def get_val_images(self):

        real_A_val = util.tensor2im(self.real_A_val.data)
        fake_B_val = util.tensor2im(self.fake_B_val.data)
        seg_fake_val = util.tensor2seg(torch.max(self.seg_fake_B_val.data,dim=1,keepdim=True)[1])
        seg_real_val = util.tensor2seg(torch.max(self.seg_real_B_val.data,dim=1,keepdim=True)[1])
        rec_A_val = util.tensor2im(self.rec_A_val.data)
        real_B_val = util.tensor2im(self.real_B_val.data)
        fake_A_val = util.tensor2im(self.fake_A_val.data)
        rec_B_val = util.tensor2im(self.rec_B_val.data)
        input_Seg_val = util.tensor2seg(self.input_Seg_val[:,1,:,:])#util.tensor2realseg(self.input_Seg_val.data)
        visual_list_val = [('real_A', real_A_val), ('fake_B', fake_B_val), ('rec_A', rec_A_val),
                                ('real_B', real_B_val), ('fake_A', fake_A_val), ('rec_B', rec_B_val),
                           ('input_seg',input_Seg_val),('fake_seg',seg_fake_val), ('real_seg',seg_real_val)]


        return  OrderedDict(visual_list_val)

    def get_val_result(self):
        self.mse_fake_A_val = torch.mean((self.real_A_val - self.fake_A_val) ** 2)
        self.mse_fake_B_val = torch.mean((self.real_B_val - self.fake_B_val) ** 2)
        self.mse_rec_A_val = torch.mean((self.real_A_val - self.rec_A_val) ** 2)
        self.mse_rec_B_val = torch.mean((self.real_B_val - self.rec_B_val) ** 2)

        self.ssim_fake_A_val = ssim(self.real_A_val, self.fake_A_val, data_range=1.0, size_average=True)
        self.ssim_fake_B_val = ssim(self.real_B_val, self.fake_B_val, data_range=1.0, size_average=True)
        self.ssim_rec_A_val = ssim(self.real_A_val, self.rec_A_val, data_range=1.0, size_average=True)
        self.ssim_rec_B_val = ssim(self.real_B_val, self.rec_B_val, data_range=1.0, size_average=True)

        self.MI_fake_A = networks.calculate_mutual_information(self.real_B_val,self.fake_A_val)
        self.MI_fake_B = networks.calculate_mutual_information(self.real_A_val,self.fake_B_val)

        self.dice_seg_real_val = self.dice_coef(self.seg_real_B_val,self.input_Seg_val)
        # print(self.mse_fake_A_val, self.mse_fake_B_val, self.mse_rec_A_val, self.mse_rec_B_val, self.ssim_fake_A_val, self.ssim_fake_B_val,
        #         self.ssim_rec_A_val, self.ssim_rec_B_val, self.dice_seg_real_val)
        return OrderedDict([('mse_fake_A',self.mse_fake_A_val), ('mse_fake_B',self.mse_fake_B_val), ('mse_rec_A',self.mse_rec_A_val),
                ('mse_rec_B',self.mse_rec_B_val), ('ssim_fake_A',self.ssim_fake_A_val), ('ssim_fake_B',self.ssim_fake_B_val),
                ('ssim_rec_A',self.ssim_rec_A_val), ('ssim_rec_B',self.ssim_rec_B_val), ('dice_seg_real',self.dice_seg_real_val),
                ('MI_fake_A',self.MI_fake_A), ('MI_fake_B',self.MI_fake_B)])

    def dice_coef(self, predicted, ground_truth, smooth=1.0):
        # print(ground_truth.shape,predicted.shape)
        ground_truth = ground_truth[:,1,:,:]
        predicted = torch.max(predicted.data, dim=1, keepdim=True)[1]#[:,1,:,:]

        intersection = torch.sum(predicted * ground_truth)
        dice = (2.0 * intersection + smooth) / (torch.sum(predicted) + torch.sum(ground_truth) + smooth)
        return dice.item()

    def save_best_model(self):
        if self.mse_fake_A_val < self.best_mse_fake_A:
            self.best_mse_fake_A = self.mse_fake_A_val
            self.save_individual_model('G_B', 'best_mse_fake_A')
        if self.mse_fake_B_val < self.best_mse_fake_B:
            self.best_mse_fake_B = self.mse_fake_B_val
            self.save_individual_model('G_A', 'best_mse_fake_B')
        if self.mse_rec_A_val < self.best_mse_rec_A:
            self.best_mse_rec_A = self.mse_rec_A_val
            self.save_individual_model('G_A', 'best_mse_rec_A')
            self.save_individual_model('G_B', 'best_mse_rec_A')
        if self.mse_rec_B_val < self.best_mse_rec_B:
            self.best_mse_rec_B = self.mse_rec_B_val
            self.save_individual_model('G_A', 'best_mse_rec_B')
            self.save_individual_model('G_B', 'best_mse_rec_B')

        if self.ssim_fake_A_val > self.best_ssim_fake_A:
            self.best_ssim_fake_A = self.ssim_fake_A_val
            self.save_individual_model('G_B', 'best_ssim_fake_A')
        if self.ssim_fake_B_val > self.best_ssim_fake_B:
            self.best_ssim_fake_B = self.ssim_fake_B_val
            self.save_individual_model('G_A', 'best_ssim_fake_B')

        if self.dice_seg_real_val > self.best_dice_seg_real:
            self.best_dice_seg_real = self.dice_seg_real_val
            self.save_individual_model('Seg_A', 'best_dice_seg_real')

        if self.MI_fake_A > self.best_MI_fake_A:
            self.best_MI_fake_A = self.MI_fake_A
            self.save_individual_model('G_B', 'best_MI_fake_A')
        if self.MI_fake_B > self.best_MI_fake_B:
            self.best_MI_fake_B = self.MI_fake_B
            self.save_individual_model('G_A', 'best_MI_fake_B')



