import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from monai.data import DataLoader, decollate_batch
from . import networks
import torch.nn.functional as F
import torch.nn as nn
import sys
import skimage
from util.visualizer import Visualizer
from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    Compose,
)





class CycleSEGModel(BaseModel):
    def name(self):
        return 'CycleSEGModel'

    def initialize(self, opt):
        self.opt = opt
        BaseModel.initialize(self, opt)
        self.list_networks = []
        self.list_networks_name = []

        self.netG_A = networks.define_G(opt)
        self.list_networks.append(self.netG_A)
        self.list_networks_name.append('G_A')

        self.netG_B = networks.define_G(opt)
        self.list_networks.append(self.netG_B)
        self.list_networks_name.append('G_B')


        self.netG_seg = networks.define_G(opt,seg_net=True)
        self.list_networks.append(self.netG_seg)
        self.list_networks_name.append('G_seg')


        if self.isTrain:
            self.netD_A = networks.define_D(opt)
            self.netD_B = networks.define_D(opt)

            self.list_networks.append(self.netD_A)
            self.list_networks_name.append('D_A')
            self.list_networks.append(self.netD_B)
            self.list_networks_name.append('D_B')

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netG_seg, 'Seg_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan= not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_seg.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            if opt.lr_enable:

                lr_scheduler_GAN_obj = eval(opt.lr_scheduler_GAN['cls'])
                lr_scheduler_discriminator_obj = eval(opt.lr_scheduler_discriminator['cls'])

                self.lr_GAN = lr_scheduler_GAN_obj(self.optimizer_G ,**opt.lr_scheduler_GAN['arg'])
                self.lr_discriminator_A = lr_scheduler_discriminator_obj(self.optimizer_D_A,
                                                                    **opt.lr_scheduler_discriminator['arg'])
                self.lr_discriminator_B = lr_scheduler_discriminator_obj(self.optimizer_D_B,
                                                                    **opt.lr_scheduler_discriminator['arg'])





    def print_network(self,visualizer_obj):
        for net, name in zip(self.list_networks,self.list_networks_name):
            visualizer_obj.print_model(net,name)


    def set_input(self, input):
        self.real_A = input['MRI'].to(self.opt.device)
        print('realA in set_input ', self.real_A.shape)
        self.real_B = input['CT'].to(self.opt.device)
        print('realB in set_input ', self.real_B.shape)
        self.real_Seg = input['label'].to(self.opt.device)
        print('real seg in set_input before onehot ', np.shape(self.real_Seg), flush=True)

        self.real_Seg  = one_hot(self.real_Seg , num_classes=self.opt.num_classes, dim=1)
        print('real seg in set_input ', self.real_Seg.shape)


    def get_networks(self):
        return self.list_networks

    def get_networks_name(self):
        return self.list_networks_name

    def eval_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A_val = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B_val = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A_val = 0
            self.loss_idt_B_val = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A_val = self.criterionGAN(pred_fake, True)
        #
        # # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B_val = self.criterionGAN(pred_fake, True)
        #
        # # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A_val = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        #
        # # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B_val = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        #
        # Segmentation loss
        self.seg_fake_B = self.netG_seg.forward(self.fake_B)
        self.loss_seg_val = self.loss_seg_function(self.seg_fake_B, self.real_Seg)




    def eval_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D


    def eval_D_A(self):
        self.loss_D_A_val = self.eval_D_basic(self.netD_A, self.real_B, self.fake_B)
        # return loss_D_A_val

    def eval_D_B(self):
        self.loss_D_B_val = self.eval_D_basic(self.netD_B, self.real_A, self.fake_A)
        # return loss_D_B_val

    def evaluation_step(self):
        self.eval_G()
        self.eval_D_A()
        self.eval_D_B()
        # return self.dic_images, loss_D_A, loss_D_B

    def train(self):
        for model in self.list_networks:
            model.train()

    def eval(self):
        for model in self.list_networks:
            model.eval()


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def backward_G(self):
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
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        #
        # # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
        #
        # # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        #
        # # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        #
        # Segmentation loss
        self.seg_fake_B = self.netG_seg.forward(self.fake_B)
        self.loss_seg = self.loss_seg_function(self.seg_fake_B, self.real_Seg)

        # combined loss

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_seg +\
                      self.loss_idt_A +self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # G_A and G_B

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        if self.opt.lr_enable:
            self.lr_GAN.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        if self.opt.lr_enable:
            self.lr_discriminator_A.step()
        # # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        if self.opt.lr_enable:
            self.lr_discriminator_B.step()
    def get_loss_name(self):
        if self.opt.identity > 0.0:
            return ['D_A','G_A','Cyc_A','idt_A',
                    'D_B','G_B','Cyc_B' ,'idt_B',
                    'Seg_B']
        else:
            return ['D_A','G_A','Cyc_A',
                    'D_B','G_B','Cyc_B',
                    'Seg_B']


    def get_current_errors(self,is_train=True):

        D_A = self.loss_D_A.item()  if is_train else self.loss_D_A_val
        G_A = self.loss_G_A.item() if is_train else self.loss_G_A_val
        Cyc_A = self.loss_cycle_A.item() if is_train else self.loss_cycle_A_val.item()
        D_B = self.loss_D_B.item() if is_train else self.loss_D_B_val
        G_B = self.loss_G_B.item() if is_train else self.loss_G_B_val
        Cyc_B = self.loss_cycle_B.item() if is_train else self.loss_cycle_B_val.item()
        Seg_B = self.loss_seg.item() if is_train else self.loss_seg_val.item()
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.item()
            idt_B = self.loss_idt_B.item()
            return {'D_A': D_A,'G_A':G_A,'Cyc_A':Cyc_A,'idt_A':idt_A,
                    'D_B':D_B,'G_B':G_B,'Cyc_B':Cyc_B ,'idt_B':idt_B,
                    'Seg_B':Seg_B}
        else:
            return {'D_A': D_A,'G_A':G_A,'Cyc_A':Cyc_A,
                    'D_B':D_B,'G_B':G_B,'Cyc_B':Cyc_B ,
                    'Seg_B':Seg_B}


    def get_current_visuals(self):

        real_A = self.real_A.data
        fake_B = self.fake_B.data
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=True)])
        seg_B = post_trans(self.seg_fake_B)
        manual_B = self.real_Seg.data
        rec_A = self.rec_A.data
        real_B = self.real_B.data
        fake_A = self.fake_A.data
        rec_B = self.rec_B.data
        if self.opt.identity > 0.0:
            idt_A = self.idt_A.data
            idt_B = self.idt_B.data
            return {'real_A': real_A, 'fake_B': fake_B, 'rec_A': rec_A, 'idt_B':idt_B,'seg_B':seg_B,
                    'real_B':real_B, 'fake_A': fake_A, 'rec_B':rec_B, 'idt_A':idt_A, 'real_seg':manual_B}
        else:
            return {'real_A': real_A, 'fake_B': fake_B, 'rec_A': rec_A, 'seg_B':seg_B,
                    'real_B':real_B, 'fake_A': fake_A, 'rec_B':rec_B,  'real_seg':manual_B}

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label)
        self.save_network(self.netD_A, 'D_A', label)
        self.save_network(self.netG_B, 'G_B', label)
        self.save_network(self.netD_B, 'D_B', label)
        self.save_network(self.netG_seg, 'Seg_B', label)

    def evaluation2_step(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = sliding_window_inference(
                inputs=self.real_B,
                roi_size=self.opt.roi_size,
                sw_batch_size=1,
                predictor=self.netG_A,
                overlap=0.5,
            )
            self.loss_idt_A_val = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = sliding_window_inference(
                inputs=self.real_A,
                roi_size=self.opt.roi_size,
                sw_batch_size=1,
                predictor=self.netG_B,
                overlap=0.5,
            )
            self.loss_idt_B_val = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A_val = 0
            self.loss_idt_B_val = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = sliding_window_inference(
            inputs=self.real_A,
            roi_size=self.opt.roi_size,
            sw_batch_size=1,
            predictor=self.netG_A,
            overlap=0.5,
        )

        # # D_B(G_B(B))
        self.fake_A = sliding_window_inference(
            inputs=self.real_B,
            roi_size=self.opt.roi_size,
            sw_batch_size=1,
            predictor=self.netG_B,
            overlap=0.5,
        )

        # # Forward cycle loss
        self.rec_A = sliding_window_inference(
            inputs=self.fake_B,
            roi_size=self.opt.roi_size,
            sw_batch_size=1,
            predictor=self.netG_B,
            overlap=0.5,
        )
        self.loss_cycle_A_val = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        #
        # # Backward cycle loss
        self.rec_B = sliding_window_inference(
            inputs=self.fake_A,
            roi_size=self.opt.roi_size,
            sw_batch_size=1,
            predictor=self.netG_A,
            overlap=0.5,
        )
        self.loss_cycle_B_val = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        #
        # Segmentation loss
        self.seg_fake_B = sliding_window_inference(
            inputs=self.fake_B,
            roi_size=self.opt.roi_size,
            sw_batch_size=1,
            predictor=self.netG_seg,
            overlap=0.5,
        )
        self.loss_seg_val = self.loss_seg_function(self.seg_fake_B, self.real_Seg)


        self.loss_G_A_val = 0

        self.loss_G_B_val = 0
        self.loss_D_A_val = 0
        self.loss_D_B_val = 0





