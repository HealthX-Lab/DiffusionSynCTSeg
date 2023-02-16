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
from util.visualizer import Visualizer
from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from monai.networks.utils import one_hot





class CycleSEGModel(BaseModel):
    def name(self):
        return 'CycleSEGModel'

    def initialize(self, opt):
        self.opt = opt
        print('first of init',flush=True)
        BaseModel.initialize(self, opt)
        self.list_networks = []
        self.list_networks_name = []

        self.netG_A = networks.define_G(opt)
        self.netG_B = networks.define_G(opt)

        self.list_networks.append(self.netG_A)
        self.list_networks_name.append('G_A')
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
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan= not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_seg.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            # if opt.lr_enable:
            #
            #     lr_scheduler_GAN_obj = eval(opt.lr_scheduler_GAN['cls'])
            #     lr_scheduler_discriminator_obj = eval(opt.lr_scheduler_discriminator['cls'])
            #
            #     self.lr_GAN = lr_scheduler_GAN_obj(self.optimizer_G ,**opt.lr_scheduler_GAN['arg'])
            #     self.lr_discriminator_A = lr_scheduler_discriminator_obj(self.optimizer_D_A,
            #                                                         **opt.lr_scheduler_discriminator['arg'])
            #     self.lr_discriminator_B = lr_scheduler_discriminator_obj(self.optimizer_D_B,
            #                                                         **opt.lr_scheduler_discriminator['arg'])

        print('end of init',flush=True)



    def print_network(self,visualizer_obj):
        for net, name in zip(self.list_networks,self.list_networks_name):
            visualizer_obj.print_model(net,name)

    def set_input(self, input):
        self.real_A = input['MRI'].to(self.opt.device)
        self.real_B = input['CT'].to(self.opt.device)
        self.real_Seg = input['label'].to(self.opt.device)
        self.real_Seg  = one_hot(self.real_Seg , num_classes=self.opt.num_classes, dim=1)
        # print('realA',np.shape(self.real_A), flush=True)
        # print('realB',np.shape(self.real_B), flush=True)
        # print('real seg',np.shape(self.real_Seg), flush=True)

    def get_networks(self):
        return self.list_networks

    def get_networks_name(self):
        return self.list_networks_name

    def evaluation_step(self):
        self.backward_G()
        self.backward_D_A()
        self.backward_D_B()

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
        print("before pool *****************", flush=True)
        # fake_B = self.fake_B_pool.query(self.fake_B)
        print(np.shape(self.fake_B))
        print("after pool ******************", flush=True)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        print("after netD_A *****************", flush=True)

    def backward_D_B(self):
        # fake_A = self.fake_A_pool.query(self.fake_A)
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
        print('*****  loss G_A  in G**', np.shape(self.loss_G_A))
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
        print('*****  loss G_B  in G**', np.shape(self.loss_G_B))
        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        print('*****  loss_cycle_A  in G**', np.shape(self.loss_cycle_A))
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        print('*****  loss_cycle_B  in G**', np.shape(self.loss_cycle_B))
        # Segmentation loss
        self.seg_fake_B = self.netG_seg.forward(self.fake_B)
        if self.opt.seg_norm == 'DiceNorm':
            self.loss_seg = self.dice_loss_seg(self.seg_fake_B, self.real_Seg)
        elif self.opt.seg_norm == 'CrossEntropy':
            self.loss_seg = self.CE_loss_seg(self.seg_fake_B, self.real_Seg_one)
        elif self.opt.seg_norm == 'DiceCELoss':
            self.loss_seg = self.DiceCELoss_seg(self.seg_fake_B, self.real_Seg_one)

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_seg
        self.loss_G.backward()

    def optimize_parameters(self):
        # G_A and G_B
        print('before G ******')
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        print('afterG ******')
        # if opt.lr_enable:
        #     self.lr_GAN.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # if opt.lr_enable:
            # self.lr_discriminator_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # if opt.lr_enable:
        #     self.lr_discriminator_B.step()

    def get_current_errors(self):
        print('*****  loss D_A  **',np.shape(self.loss_D_A))
        print('*****  loss D_B  **', np.shape(self.loss_D_B))
        print('*****  loss G_A  **', np.shape(self.loss_G_A))
        print('*****  loss G_B  **', np.shape(self.loss_G_B))
        print('*****  loss seg_B  **', np.shape(self.loss_seg))
        D_A = self.loss_D_A.item()
        G_A = self.loss_G_A.item()
        Cyc_A = self.loss_cycle_A.item()
        D_B = self.loss_D_B.item()
        G_B = self.loss_G_B.item()
        Cyc_B = self.loss_cycle_B.item()
        Seg_B = self.loss_seg.item()
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

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        seg_B = util.tensor2seg(torch.max(self.seg_fake_B.data,dim=1,keepdim=True)[1])
        manual_B = util.tensor2seg(torch.max(self.real_Seg.data,dim=1,keepdim=True)[1])
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
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
        self.save_network(self.netG_seg, 'Seg_A', label)

    # def update_learning_rate(self):
    #     lrd = self.opt.lr / self.opt.niter_decay
    #     lr = self.old_lr - lrd
    #     for param_group in self.optimizer_D_A.param_groups:
    #         param_group['lr'] = lr
    #     for param_group in self.optimizer_D_B.param_groups:
    #         param_group['lr'] = lr
    #     for param_group in self.optimizer_G.param_groups:
    #         param_group['lr'] = lr
    #
    #     print('update learning rate: %f -> %f' % (self.old_lr, lr))
    #     self.old_lr = lr