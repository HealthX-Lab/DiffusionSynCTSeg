import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from collections import OrderedDict
import itertools
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from pytorch_msssim import ssim
import torch.nn as nn
from torch.autograd import Variable


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    
    def name(self):
        return 'cut'
    @staticmethod
    def modify_commandline_options(opt, is_train=True):
        """  Configures options specific for CUT model
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_nc', type=int, default=1,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'],
                            help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks',
                            choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2',
                                     'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'],
                            help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'],
                            help='instance normalization or batch normalization for D')
        parser.add_argument('--init_type', type=str, default='xavier',
                            choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=False,help='no dropout for the generator')
        

        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--no_antialias', action='store_true',
                            help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true',
                            help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')

        parser.add_argument('--segmentation', type=bool, default=True, help='adding segmentation model')

        parser.add_argument('--seg_norm', type=str, default='CrossEntropy',
                            help='DiceNorm or CrossEntropy or CombinationLoss')
        parser.add_argument('--seg_weight', type=float, default=2.0,
                            help='segmentation loss weight')
        parser.add_argument('--separate_seg', type=bool, default=False,
                            help='separate segmentation and translation')
        parser.add_argument('--mri_ct_seg', type=bool, default=False,
                            help='train segmentation with mri and ct')
        parser.add_argument('--add_quantum_noise', type=bool, default=False,
                            help='add quantum noise to fake ct images before segmentaiton')
        parser.add_argument('--mode_seg', type=str, default='2d',
                            help='2d or 3d segmentation model ')
        parser.add_argument('--Depth', type=int, default=32, help='# depth for 3D segmentation')
        

        parser.set_defaults(pool_size=0)  # no image pooling
        new_opts, _ = parser.parse_known_args() 

        # Now you might want to merge new_opts into opt
        for key, value in vars(new_opts).items():
            setattr(opt, key, value)
        opt.crossentropy_weight = [1, 30]


        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        
        return opt

    def initialize(self, opt):

        self.image_array = np.zeros((30, 136, 156), dtype=np.float32)

        opt = self.modify_commandline_options(opt)
        BaseModel.initialize(self, opt)
        self.Tensor = torch.cuda.FloatTensor
        nb = opt.batchSize
        size = opt.fineSize
        self.input_Seg = self.Tensor(nb, opt.output_nc_seg, size, size)
        self.results = {}
        

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']
        if opt.segmentation:
            self.loss_names += ['seg']
            self.visual_names += ['seg']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        if opt.segmentation:
            self.model_names += ['Seg']
        if opt.boundry_loss:
            self.loss_names += ['boundry']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if opt.segmentation:
            self.netSeg = networks.define_seg(opt.input_nc_seg, opt.output_nc_seg,
                                              opt.which_model_netSeg, opt.norm, opt.init_type, not opt.no_dropout,
                                              self.gpu_ids, uncertainty=opt.uncertainty,mode=opt.mode_seg)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            if opt.boundry_loss:
                self.surface_loss = networks.SurfaceLoss(idc=[1])

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            if not opt.segmentation or (opt.segmentation and opt.separate_seg):
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_G)
                if opt.segmentation:
                    self.optimizer_seg = torch.optim.Adam(self.netSeg.parameters(), lr=opt.lr,
                                                          betas=(opt.beta1, opt.beta2))
                    self.optimizers.append(self.optimizer_seg)
            if opt.segmentation and not opt.separate_seg:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(),self.netSeg.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_G)

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_D)
            if opt.mode_seg == '3d':
                arr = np.array(opt.crossentropy_weight)
                weight = torch.from_numpy(arr).cuda().float()
                self.criterion_seg = nn.CrossEntropyLoss(weight=weight)
                Depth = opt.Depth
                nb = opt.batchSize
                size = opt.fineSize
                self.input_Seg_one = self.Tensor(size=(nb,  Depth, size, size))
                self.fake_B_3d = self.Tensor(nb, 1, Depth, size, size)
                self.real_B_3d = self.Tensor(nb, 1, Depth, size, size)
                self.input_B = self.Tensor(nb, 1, Depth, size, size)

        if self.opt.yh_run_model == 'Test':
            if opt.mode_seg == '3d':
                Depth = opt.Depth
                nb = opt.batchSize
                size = opt.fineSize
                mc_number = self.opt.num_samples_uncertainty
                self.input_Seg_one = self.Tensor(size=(nb,  Depth, size, size))
                self.fake_B_3d = self.Tensor(mc_number, 1, Depth, size, size)
                self.real_B_3d = self.Tensor(nb, 1, Depth, size, size)
                self.input_B = self.Tensor(nb, 1, Depth, size, size)

                self.fake_seg = self.Tensor(mc_number, 2, Depth, size, size)
                self.real_seg = self.Tensor(mc_number, 2, Depth, size, size)
            self.image_array = np.zeros((30, 136, 156), dtype=np.float32)

            self.load_network_seg(self.netG, 'G', opt.which_epoch)
            self.load_network_seg(self.netSeg, 'Seg', opt.which_epoch)
            self.mse_A = 0
            self.mse_B = 0
            self.mse_fake_A = 0
            self.mse_fake_B = 0
            self.seg_fake = []
            self.dice_seg_real = []
            self.seg_real_IOU = 0
            self.var_fake_A = 0
            self.var_fake_B = 0
            self.var_seg_real = 0
            self.var_seg_fake = 0
            self.number_of_images = -1
            self.precision = 0
            self.recall = 0
            self.specificity = 0

            self.ssim_value_A = 0
            self.ssim_value_B = 0
            self.TP = 0
            self.TN = 0
            self.FP = 0
            self.FN = 0

    def set_zero(self):

        self.mse_B = 0
        self.mse_fake_B = 0

        self.seg_real_IOU = 0
        self.var_fake_A = 0
        self.var_fake_B = 0
        self.var_seg_real = 0
        self.var_seg_fake = 0
        self.number_of_images = self.number_of_images + 1
        self.precision = 0
        self.recall = 0
        self.specificity = 0

        self.ssim_value_B = 0
        if self.number_of_images % 41 == 0:
            self.TP = 0
            self.TN = 0
            self.FP = 0
            self.FN = 0
            self.number_of_images = 0
            self.dice = 0
            self.iou = 0
            self.recal = 0
            self.specificity = 0
            self.percision = 0
            self.seg_fake = []
            self.dice_seg_real = []
            self.image_array = np.zeros((30, 136, 156), dtype=np.float32)


    def data_dependent_initialize(self, data, data2=None):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.segmentation and self.opt.separate_seg and self.opt.mode_seg=='2d':
                self.loss_seg.backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad()
        if self.opt.segmentation and self.opt.separate_seg and self.opt.mode_seg=='2d'or (self.opt.mode_seg=='3d'and self.data_number%41 ==40):
            self.optimizer_seg.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
        if (self.opt.segmentation and self.opt.separate_seg and self.opt.mode_seg=='2d')or (self.opt.mode_seg=='3d'and self.data_number%41 ==40 and self.data_number ):
            self.loss_seg.backward()
            self.optimizer_seg.step()

    def set_input(self, input, data2=None,d=0):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_paths_A = input['A_paths']
        self.image_paths_B = input['B_paths']
        self.image_paths_seg = input['Seg_paths']
        self.dist_map_label = input['dist_map']
        if self.opt.mode_seg == '2d':
            if self.opt.segmentation:
                from torch.autograd import Variable
                self.real_Seg = input['Seg_one'].to(self.device)
                self.real_Seg = Variable(self.real_Seg.long())
                input_Seg = input['Seg']
                self.input_Seg.resize_(input_Seg.size()).copy_(input_Seg)
                self.input_Seg = Variable(self.input_Seg)
        elif self.opt.mode_seg =='3d':
            self.data_number = d
            seg_2D = input['Seg_one']

            if d%41 < self.opt.Depth:
                # print('in setting 3D seg in input ', d%41)
                self.input_Seg_one[:, d%41, :, :] = seg_2D
                self.real_B_3d[:, :, d%41, :, :] = self.real_B

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        if self.opt.segmentation and self.opt.isTrain:
            # fake = self.fake_B.detach()
            if self.opt.segmentation and self.opt.separate_seg:
                fake = self.fake_B.detach()
                if self.opt.mode_seg == '2d':
                    if self.opt.add_quantum_noise:
                        fake = self.add_poisson_noise(fake)
                    self.seg_fake_B = self.netSeg(fake)
                elif self.opt.mode_seg == '3d':
                    d = self.data_number
                    if d % 41 < self.opt.Depth:
                        # print(' in setting 3d fake B')
                        self.fake_B_3d[:, :, d % 41, :, :] = fake
                    if self.data_number % 41 == 40 and self.data_number:
                        # print(' in segmentaion 3d')
                        self.fake_seg = self.netSeg(self.fake_B_3d)

            else:
                if self.opt.add_quantum_noise:
                    self.fake_B = self.add_poisson_noise(self.fake_B)
                self.seg_fake_B = self.netSeg(self.fake_B)
                if self.opt.mode_seg == '3d':
                    d = self.data_number
                    if d % 41 < self.opt.Depth:
                        # print(' in setting 3d fake B')
                        fake = self.fake_B
                        self.fake_B_3d[:, :, d % 41, :, :] = fake
                    if self.data_number % 41 == 40 and self.data_number:
                        # print(' in segmentaion 3d')
                        self.fake_seg = self.netSeg(self.fake_B_3d)

        if self.opt.eval:
            self.seg_real_B = self.netSeg(self.real_B)
            self.seg_fake_B = self.netSeg(self.fake_B)
        if self.opt.mri_ct_seg:
            self.seg_real_A = self.netSeg(self.real_A)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both

        if self.opt.segmentation:
            if self.opt.mode_seg == '2d':
                arr = np.array(self.opt.crossentropy_weight)
                weight = torch.from_numpy(arr).cuda().float()
                self.loss_seg = networks.CrossEntropy2d(self.seg_fake_B, self.real_Seg, weight=weight)
                if self.opt.mri_ct_seg:
                    self.loss_seg += networks.CrossEntropy2d(self.seg_real_A, self.real_Seg, weight=weight)
                    self.loss_seg = self.loss_seg / 2.0
                if self.opt.boundry_loss:
                    self.loss_boundry = self.surface_loss(F.softmax(self.seg_fake_B, dim=1),
                                                              self.dist_map_label.cuda()) / 10.0
                    b_weight = self.opt.boundry_loss_weight
                    self.loss_seg = self.loss_seg + self.loss_boundry*b_weight
            elif self.opt.mode_seg == '3d':
                if self.data_number % 41 == 40 and self.data_number:
                    self.real_Seg = Variable(self.input_Seg_one.long())
                    self.loss_seg = self.criterion_seg(self.fake_seg, self.real_Seg)

            if not self.opt.separate_seg:
                self.loss_G += (self.loss_seg * self.opt.seg_weight)



        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        self.visual_real_A = util.tensor2im(self.real_A.data)
        self.visual_fake_B = util.tensor2im(self.fake_B.data)

        # if self.opt.nce_idt:
        #     self.visual_idt_B = util.tensor2im(self.idt_B)

        if self.opt.segmentation and self.opt.mode_seg =='2d':
            self.visual_fake_seg = util.tensor2seg(F.softmax(self.seg_fake_B.detach().cpu(), dim=1)[:,1,:,:])
        elif self.opt.mode_seg =='3d':
            self.visual_fake_seg = util.tensor2seg(F.softmax(self.fake_seg.detach().cpu(), dim=1)[:,1,:,:])


        if self.opt.eval:
            self.visual_real_seg = util.tensor2seg(F.softmax(self.seg_real_B.detach().cpu(), dim=1)[:,1,:,:])



    

    def get_current_visuals(self):
        if self.opt.phase != 'test':
            return_images = []
            return_images += [('real_A', self.visual_real_A)]
            return_images += [('fake_B', self.visual_fake_B)]

            # if self.opt.nce_idt:
            #     return_images += [('idt', self.visual_idt_B)]

            if self.opt.segmentation:
                return_images += [('fake_seg', self.visual_fake_seg)]
                if self.opt.eval:
                    return_images += [('real_seg', self.visual_real_seg)]

            return OrderedDict(return_images)
        elif self.opt.phase == 'test':
            visuals = {}

            visuals['fake_B'] = [util.tensor2im(self.fake_B.data)]
            visuals['fake_seg'] = [util.tensor2seg(F.softmax(self.seg_fake_B.data, dim=1)[:, 1, :, :])]
            visuals['seg_real'] = [util.tensor2seg(F.softmax(self.seg_real_B.data, dim=1)[:,1,:,:])]#>0.005

            heatmap = {key: util.tensor2seg(self.heatmap[key]) for key in self.heatmap.keys() if 'seg' in key}
            uncertainty_map = {key: util.tensor2map(self.uncertainty_map[key]) for key in self.uncertainty_map.keys()}
            confidence_map = {key: util.tensor2map(self.confidence_map[key]) for key in self.confidence_map.keys() if
                              'seg' in key}
            entropy_map = {key: util.tensor2map(self.entropy_map[key]) for key in self.entropy_map.keys() if
                           'seg' in key}
            var_map = {key: util.tensor2map(self.var[key]) for key in self.var.keys() if 'seg' not in key}

            real_A = util.tensor2im(self.real_A.data)
            real_B = util.tensor2im(self.real_B.data)
            input_Seg = util.tensor2realseg(self.input_Seg.data)

            return OrderedDict([
                ('real_A', real_A),
                ('real_B', real_B),
                ('input_seg', input_Seg),
                ('var_map', var_map),
                ('Heatmap', heatmap),
                ('Uncertainty_Map', uncertainty_map),
                ('Confidence_Map', confidence_map),
                ('Entropy_Map', entropy_map),
                ('Visuals', visuals),
                ('seg_gamma1', util.tensor2seg(self.visual_gamma1)),
                ('seg_gamma2', util.tensor2seg(self.visual_gamma2))
            ])

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        models = [self.netG, self.netSeg]
        for i in range(0,len(models)):
            for m in models[i].modules():
                if m.__class__.__name__.startswith('Dropout'):
                    # print('******', m.__class__.__name__)
                    m.train()
    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        if not self.opt.MC_uncertainty:
            with torch.no_grad():
                self.forward()
                self.compute_visuals()
        elif self.opt.MC_uncertainty :

            if self.opt.mode_seg == '2d':
                self.create_uncertainty()
            elif self.opt.mode_seg == '3d':
                for i in range(0, self.opt.num_samples_uncertainty):

                    if self.data_number % 41 <= (self.opt.Depth - 1):
                        self.forward()
                        d = self.data_number
                        self.fake_B_3d[i, 0, d % 41, :, :] = self.fake_B

                if self.data_number % 41 == 40 and self.data_number:
                    self.compute_3D()

    def compute_3D(self):

        for i in range(0, self.opt.num_samples_uncertainty):

            self.fake_seg[i, :, :, :, :] = F.softmax(self.netSeg.forward(self.fake_B_3d[i].unsqueeze(0)), dim=1).detach().cpu()
            self.real_seg[i, :, :, :, :] = F.softmax(self.netSeg.forward(self.real_B_3d), dim=1).detach().cpu()

        self.result_list = []
        for slice_number in range(0, self.opt.Depth):
            self.MC_uncertainty_outputs = {'fake_B': [], 'fake_seg': [], 'seg_real': []}

            for i in range(0, self.opt.num_samples_uncertainty):
                fake_seg = self.fake_seg[i, :, slice_number, :, :].unsqueeze(0)
                real_seg = self.real_seg[i, :, slice_number, :, :].unsqueeze(0)
                fake_B = self.fake_B_3d[i, :, slice_number, :, :].unsqueeze(0)
                self.MC_uncertainty_outputs['fake_seg'].append(fake_seg.detach().cpu())
                self.MC_uncertainty_outputs['seg_real'].append(real_seg.detach().cpu())
                self.MC_uncertainty_outputs['fake_B'].append(fake_B.detach().cpu())

            self.number_of_images = slice_number
            self.mean, self.var, self.heatmap, self.uncertainty_map, self.confidence_map, self.entropy_map = self.compute_txt(
                self.MC_uncertainty_outputs)

            slice_seg = self.input_Seg_one[ :, slice_number, :, :].detach().cpu()
            self.input_Seg[:,1,:,:] = slice_seg
            self.real_B = self.real_B_3d[:, :, slice_number, :, :].detach().cpu()
            self.set_coef()
            result = self.get_coef()
            result['epoch'] = self.opt.which_epoch
            result['data_number'] = slice_number
            self.result_list.append(result)





    def create_uncertainty(self):
        self.MC_uncertainty_outputs = {'fake_B': [], 'fake_seg': [], 'seg_real': []}
        self.enable_dropout()

        for i in range(0, self.opt.num_samples_uncertainty):
            self.forward()

            fake_seg = F.softmax(self.seg_fake_B.data, dim=1)
            real_seg = F.softmax(self.seg_real_B.data, dim=1)
            fake_B = self.fake_B
            self.MC_uncertainty_outputs['fake_seg'].append(fake_seg.detach().cpu())
            self.MC_uncertainty_outputs['seg_real'].append(real_seg.detach().cpu())
            self.MC_uncertainty_outputs['fake_B'].append(fake_B.detach().cpu())

        self.mean, self.var, self.heatmap, self.uncertainty_map, self.confidence_map, self.entropy_map = self.compute_txt(
            self.MC_uncertainty_outputs)
        self.input_Seg = self.input_Seg.detach().cpu()
        self.real_B = self.real_B.detach().cpu()
        if self.number_of_images < 30:
            # print('image number in set image', self.number_of_images)
            self.set_image(self.number_of_images, self.crop_images(self.mean['seg_real']))
        self.set_coef()
        if self.number_of_images % 41 == 40:
            # print('image number in 3D image', self.number_of_images)
            self.set_3Dcoef()

    def compute_txt(self, outputs):
        mean = {key: None for key in outputs.keys()}
        var = {key: None for key in outputs.keys()}
        heatmap = {key: None for key in outputs.keys()}
        uncertainty_map = {key: None for key in outputs.keys()}
        confidence_map = {key: None for key in outputs.keys()}
        entropy_map = {key: None for key in outputs.keys()}
        for key in outputs.keys():

            if 'seg' in key:
                output_probability = torch.stack(outputs[key])
                mean[key] = output_probability.mean(dim=0)
                var[key] = output_probability.var(dim=0)
                heatmap[key] = torch.argmax(mean[key], dim=1)
                self.heat = heatmap[key]
                uncertainty_map[key] = var[key].max(dim=1)[0].sqrt()
                confidence_map[key] = mean[key].max(dim=1)[0]
                entropy_map[key] = self.compute_entropy(mean[key])
                if key == 'seg_real':
                    self.cal_manual_dice(outputs[key])
                    self.cal_manual_iou(outputs[key])


            else:
                stacked_output = torch.stack(outputs[key])
                mean[key] = stacked_output.mean(dim=0)
                var[key] = stacked_output.var(dim=0)
                uncertainty_map[key] = var[key].sqrt()[0]

        return mean, var, heatmap, uncertainty_map, confidence_map, entropy_map

    def cal_manual_dice(self, segmentation):
        mean_seg = torch.stack(segmentation).mean(dim=0)
        one_seg = segmentation[0]
        self.mean_pure_dice, self.mean_dice_gamma_1, self.mean_dice_gamma_2 = self.preprocessed_dice(mean_seg)
        self.one_pure_dice, self.one_dice_gamma_1, self.one_dice_gamma_2 = self.preprocessed_dice(one_seg)

    def cal_manual_iou(self, segmentation):
        mean_seg = torch.stack(segmentation).mean(dim=0)
        one_seg = segmentation[0]
        self.mean_pure_iou, self.mean_iou_gamma_1, self.mean_iou_gamma_2 = self.preprocessed_iou(mean_seg)
        self.one_pure_iou, self.one_iou_gamma_1, self.one_iou_gamma_2 = self.preprocessed_iou(one_seg)

    def preprocessed_iou(self, seg):
        pure_iou = self.get_IOU(torch.argmax(seg, dim=1), self.input_Seg)
        seg_gamma_1 = self.apply_gamma(seg, 0.1)
        iou_gamma_1 = self.get_IOU(torch.argmax(seg_gamma_1, dim=1), self.input_Seg)
        seg_gamma_2 = self.apply_gamma(seg, 0.2)
        iou_gamma_2 = self.get_IOU(torch.argmax(seg_gamma_2, dim=1), self.input_Seg)
        return pure_iou, iou_gamma_1, iou_gamma_2

    def preprocessed_dice(self, seg):
        pure_dice = self.dice_coef(torch.argmax(seg, dim=1), self.input_Seg, smooth=1)
        seg_gamma_1 = self.apply_gamma(seg, 0.1)
        y_pred1 = torch.argmax(seg_gamma_1, dim=1)
        dice_gamma_1 = self.dice_coef(y_pred1, self.input_Seg, smooth=1)
        seg_gamma_2 = self.apply_gamma(seg, 0.2)
        y_pred2 = torch.argmax(seg_gamma_2, dim=1)
        dice_gamma_2 = self.dice_coef(y_pred2, self.input_Seg, smooth=1)
        self.visual_gamma1 = torch.argmax(seg_gamma_1, dim=1)
        self.visual_gamma2 = torch.argmax(seg_gamma_2, dim=1)
        self.processed_dice3d(y_pred1, y_pred2)
        return pure_dice, dice_gamma_1, dice_gamma_2

    def processed_dice3d(self, y_pred1, y_pred2):
        y_true = self.input_Seg[:, 1, :, :].view(-1).detach().cpu().numpy()
        y_pred1 = y_pred1.view(-1).detach().cpu().numpy()
        y_pred2 = y_pred2.view(-1).detach().cpu().numpy()

        self.tn1, self.fp1, self.fn1, self.tp1 = confusion_matrix(y_true, y_pred1, labels=[0, 1]).ravel()
        self.tn2, self.fp2, self.fn2, self.tp2 = confusion_matrix(y_true, y_pred2, labels=[0, 1]).ravel()



    def crop_images(self, seg):
        height, width = seg.shape[2:]  # In case of color image, ignore the color channel
        left = 50
        right = width - 50
        top = 60
        bottom = height - 60
        cropped_image = seg[0, 1, top:bottom, left:right]
        return cropped_image

    def pad_image(self, seg, cropped_image):
        # Assuming seg is a 4D tensor with shape [N, C, H, W]
        height, width = seg.shape[2:]  # Get height and width

        # Define crop boundaries
        left, right = 50, width - 50
        top, bottom = 60, height - 60

        # Padded image (zero-filled)
        padded_image = torch.zeros_like(seg)
        padded_image[:, 1, top:bottom, left:right] = cropped_image

        return padded_image

    def apply_gamma(self, seg, gamma):
        from skimage.filters import threshold_otsu
        from skimage import exposure

        cropped_image = self.crop_images(seg)
        cropped_image = self.pad_image(seg, cropped_image)

        if isinstance(cropped_image, torch.Tensor):
            cropped_image = cropped_image.detach().cpu().numpy()  # Convert to numpy
            cropped_image = cropped_image.astype(np.float32)

        cropped_image = exposure.adjust_gamma(cropped_image, gamma)
        non_zero_cropped_image = cropped_image[cropped_image > 0]
        threshold_value = 0
        if len(non_zero_cropped_image) > 0:  # Check if there are non-zero values
            threshold_value = threshold_otsu(non_zero_cropped_image)
        threshold_value = threshold_value - 0.15
        binary_image = cropped_image > threshold_value

        binary_image = torch.from_numpy(binary_image)
        binary_image = binary_image.int()

        return binary_image

    def probability(self,img):
        stacked_output = torch.stack(img)
        mean = stacked_output.mean(dim=0)
        var = stacked_output.var(dim=0)
        return mean, var

    def calculate_ssim(self, image1, image2):
        import torch
        import torchmetrics
        import torchvision.transforms.functional as F

        sigma = 2
        kernel_size = 7

        # Applying Gaussian blur
        image1 = F.gaussian_blur(image1, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        image2 = F.gaussian_blur(image2, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        # Initialize SSIM metric
        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(kernel_size=11, data_range=1.0,
                                                                    gaussian_kernel=True, sigma=1)

        # Calculate SSIM
        ssim_value = ssim_metric(image1, image2)
        return ssim_value.item()

    def set_coef(self):
        self.var_fake_B_2D = self.avg_uncertainty(self.uncertainty_map['fake_B'])
        self.var_seg_real_2D = self.avg_uncertainty(self.uncertainty_map['seg_real'])
        self.var_seg_fake_2D = self.avg_uncertainty(self.uncertainty_map['fake_seg'])
        self.ssim_value_B_2D = ssim(self.mean['fake_B'], self.real_B, data_range=1.0,
                                    size_average=True).item()
        self.new_ssim = self.calculate_ssim(self.mean['fake_B'], self.real_B)
        self.mse_fake_B_2D = self.cal_mse(self.mean['fake_B'], self.real_B)
        self.seg_real_IOU_2D = self.get_IOU(self.heatmap['seg_real'], self.input_Seg)
        self.seg_fake_2D = self.dice_coef(self.heatmap['fake_seg'], self.input_Seg, smooth=1)
        self.seg_real_2D = self.dice_coef(self.heatmap['seg_real'], self.input_Seg, smooth=1)
        self.get_confusion_matrix(self.heatmap['seg_real'], self.input_Seg, 'real')
        self.recal_2D = self.cal_recal()
        self.specificity_2D = self.cal_specificity()
        self.percision_2D = self.cal_percision()

        self.hist_similarity = self.compare_histograms(self.mean['fake_B'], self.real_B)

    def get_3dcoef(self):
        return self.result_list
    def get_coef(self):

        results = \
            {
                'mean_pure_iou': self.mean_pure_iou.item(),
                'mean_iou_gamma_1': self.mean_iou_gamma_1.item(),
                'mean_iou_gamma_2': self.mean_iou_gamma_2.item(),
                'one_pure_iou': self.one_pure_iou.item(),
                'one_iou_gamma_1': self.one_iou_gamma_1.item(),
                'one_iou_gamma_2': self.one_iou_gamma_2.item(),

                'mean_pure_dice': self.mean_pure_dice,
                'mean_dice_gamma_1': self.mean_dice_gamma_1,
                'mean_dice_gamma_2': self.mean_dice_gamma_2,
                'one_pure_dice': self.one_pure_dice,
                'one_dice_gamma_1': self.one_dice_gamma_1,
                'one_dice_gamma_2': self.one_dice_gamma_2,

                # 'mse_fake_B': self.mse_fake_B,
                'mse_fake_B_2D': self.mse_fake_B_2D,
                # 'seg_real_IOU_H': self.seg_real_IOU,
                'seg_real_IOU_2D': self.seg_real_IOU_2D.item(),
                'dice_seg_fake_2D': self.seg_fake_2D,
                'dice_seg_real_2D': self.seg_real_2D,

                'ssim_value_B_2D': self.ssim_value_B_2D,
                'new_ssim': self.new_ssim,
                # 'var_seg_fake': self.var_seg_fake,
                'avg_var_seg_fake_2D': self.var_seg_fake_2D,

                # 'var_seg_real': self.var_seg_real,
                'avg_var_seg_real_2D': self.var_seg_real_2D,

                # 'var_fake_B': self.var_fake_B,
                'avg_var_fake_B_2D': self.var_fake_B_2D,
                'corr_hist': self.hist_similarity,

                'percision_2D': self.percision_2D,
                'specificity_2D': self.specificity_2D,
                'recal_2D': self.recal_2D,

                'tp': self.tp,
                'tn': self.tn,
                'fp': self.fp,
                'fn': self.fn,
                'tp1': self.tp1,
                'tn1': self.tn1,
                'fp1': self.fp1,
                'fn1': self.fn1,
                'tp2': self.tp2,
                'tn2': self.tn2,
                'fp2': self.fp2,
                'fn2': self.fn2,

                'dice': self.dice,
                'iou': self.iou,
                'recall': self.recal,
                'specificity': self.specificity,
                'percision': self.percision,
                'CT_name': self.image_paths_B[0].split('/')[-1],
                # 'mri_name': self.image_paths_A[0].split('/')[-1],
                # 'Seg': self.image_paths_seg[0].split('/')[-1],
            }
        return results

    def dice_coef(self, predicted, ground_truth, smooth=1.0):
        ground_truth = ground_truth[:, 1, :, :]
        ground_truth = ground_truth.detach().cpu()
        intersection = torch.sum(predicted * ground_truth)
        dice = (2.0 * intersection + smooth) / (torch.sum(predicted) + torch.sum(ground_truth) + smooth)
        return dice.item()

    def get_IOU(self, predicted, ground_truth):
        ground_truth = ground_truth[:, 1, :, :]
        ground_truth = ground_truth.detach().cpu()

        intersection = torch.sum(predicted * ground_truth)
        union = (predicted.bool() | ground_truth.bool()).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    def cal_mse(self, input, target):
        target = target.detach().cpu()
        mse = torch.mean((input - target) ** 2)
        return mse.item()


    def set_3Dcoef(self):

        if 2 * self.TP + self.FP + self.FN:
            self.dice = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
        if self.TP + self.FP + self.FN:
            self.iou = self.TP / (self.TP + self.FP + self.FN)
        if self.TP + self.FN:
            self.recal = self.TP / (self.TP + self.FN)
        if self.TN + self.FP:
            self.specificity = self.TN / (self.TN + self.FP)
        if self.TP + self.FP:
            self.percision = self.TP / (self.TP + self.FP)

    def cal_recal(self):
        recal = None
        if self.tp + self.fn != 0:
            recal = self.tp / (self.tp + self.fn)
        return recal

    def cal_specificity(self):
        specificity = None
        if self.tn + self.fp != 0:
            specificity = self.tn / (self.tn + self.fp)
        return specificity

    def cal_percision(self):
        percision = None
        if self.tp + self.fp != 0:
            percision = self.tp / (self.tp + self.fp)
        return percision

    def compute_entropy(self, probs):
        epsilon = 1e-5  # To avoid log(0)
        return -(probs * torch.log(probs + epsilon)).sum(dim=1)

    def avg_uncertainty(self, uncertainty_per_pixel):
        total_uncertainty = np.mean(uncertainty_per_pixel.detach().cpu().numpy())
        return total_uncertainty

    def get_name(self):
        return self.image_paths_A[0].split('/')[-1]

    def get_image(self):
        return self.image_array

    def set_image(self, i, seg):
        self.image_array[i] = seg

    def get_confusion_matrix(self, y_pred, y_true, key):
        y_true = y_true[:, 1, :, :].detach().cpu().numpy()
        img1 = y_true  # .detach().cpu().numpy()

        img1 = img1.flatten()

        y_true = img1  # y_true.view(-1)#.detach().cpu().numpy()
        y_pred = y_pred.view(-1).detach().cpu().numpy()
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        self.TP = self.TP + self.tp
        self.TN = self.TN + self.tn
        self.FP = self.FP + self.fp
        self.FN = self.FN + self.fn

    def compare_histograms(self, tensor1, tensor2, bins=256):
        # Convert tensors to numpy arrays
        array1 = tensor1.detach().cpu().numpy()
        array2 = tensor2.detach().cpu().numpy()

        # Flatten the arrays if they are images
        array1 = array1.flatten()
        array2 = array2.flatten()

        # Compute histograms
        hist1, _ = np.histogram(array1, bins=bins, range=(-1, 1))
        hist2, _ = np.histogram(array2, bins=bins, range=(-1, 1))

        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)

        # Compare using a method (e.g., correlation)
        similarity = np.correlate(hist1, hist2)  # You can also use other methods like chi-square here

        return similarity
