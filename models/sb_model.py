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
from torch.autograd import Variable
import torch.nn as nn


class SBModel(BaseModel):
    def name(self):
        return 'sb'
    @staticmethod
    def modify_commandline_options(opt, is_train=True):
        """  Configures options specific for SB model

        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', type=str, default="sb", choices='(FastCUT, fastcut, sb)')
        

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_SB', type=float, default=1, help='weight for SB loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=True,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        # diffusion
        parser.add_argument('--num_timesteps', type=int, default=5,
                                 help='# of discrim filters in the first conv layer')
        parser.add_argument('--embedding_dim', type=int, default=1,
                                 help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--netD', type=str, default='basic_cond',
                                 choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'],
                                 help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netE', type=str, default='basic_cond',
                                 choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2',
                                          'patchstylegan2'],
                                 help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks_cond',
                                 choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2',
                                          'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
        parser.add_argument('--embedding_type', type=str, default='positional', choices=['fourier', 'positional'],
                                 help='specify generator architecture')
        parser.add_argument('--n_mlp', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'],
                                 help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'],
                                 help='instance normalization or batch normalization for D')
        parser.add_argument('--init_type', type=str, default='xavier',
                                 choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02,
                                 help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--std', type=float, default=0.25, help='Scale of Gaussian noise added to data')
        parser.add_argument('--tau', type=float, default=0.01, help='Entropy parameter')
        parser.add_argument('--no_antialias', action='store_true',
                                 help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true',
                                 help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')

        parser.add_argument('--segmentation', type=bool, default=True, help='adding segmentation model')
        # parser.add_argument('--seg_type', type=str, default='unet_128', help='unet_128,  resnet_9blocks, R2AttU_Net, AttU_Net, R2U_Net selects model to do segmentation netSeg')
        parser.add_argument('--seg_norm', type=str, default='CrossEntropy',
                                 help='DiceNorm or CrossEntropy or CombinationLoss')
        parser.add_argument('--seg_weight', type=int, default=1,
                            help='segmentation loss weight')
        parser.add_argument('--separate_seg', type=bool, default=False,
                            help='segmentation loss weight')
        parser.add_argument('--mri_ct_seg', type=bool, default=False,
                            help='segmentation loss weight')
        parser.add_argument('--add_quantum_noise', type=bool, default=False,
                            help='add quantum noise to fake ct images before segmentaiton')
        parser.add_argument('--txt', type=bool, default=True,
                            help='write info in txt file ')
        parser.add_argument('--mode_seg', type=str, default='2d',
                            help='2d or 3d segmentation model ')
        parser.add_argument('--Depth', type=int, default=32, help='# depth for 3D segmentation')
        parser.add_argument('--Generation_step_seg', type=bool, default=False,
                            help='write info in txt file ')
        parser.add_argument('--diff_uncertainty', type=bool, default=True,
                            help='write info in txt file ')


        # diffusion


        parser.set_defaults(pool_size=0)  # no image pooling

        new_opts, _ = parser.parse_known_args()

        # Now you might want to merge new_opts into opt
        for key, value in vars(new_opts).items():
            setattr(opt, key, value)
        opt.crossentropy_weight = [1, 30]
        opt.nce_idt = True
        opt.lambda_NCE=1.0

        # Set default parameters for CUT and FastCUT
        # if opt.mode.lower() == "sb":
        #     opt.nce_idt=True
        #     opt.lambda_NCE=1.0
        # elif opt.mode.lower() == "fastcut":
        #     opt.nce_idt=False
        #     opt.lambda_NCE=10.0
        #     opt.flip_equivariance=True
        #     opt.n_epochs=150
        #     opt.n_epochs_decay=50
        # 
        # else:
        #     raise ValueError(opt.mode)

        return opt

    def initialize(self, opt):
        self.Tensor = torch.cuda.FloatTensor
        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        # self.input_Seg = self.Tensor(nb, opt.output_nc_seg, size, size)
        self.input_Seg = torch.zeros(nb, opt.output_nc_seg, size, size)

        # self.image_array = np.zeros((25, 136, 156), dtype=np.float32)


        self.results = {}

        opt = self.modify_commandline_options(opt)


        opt.no_dropout = False
        BaseModel.initialize(self, opt)


        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'SB']
        self.visual_names = ['real_A', 'real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE + 1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if opt.segmentation:
            self.loss_names += ['seg']
            self.visual_names += ['seg']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'E']
        else:  # during test time, only load G
            self.model_names = ['G']
        if opt.segmentation:
            self.model_names += ['Seg']
        if opt.boundry_loss:
            self.loss_names += ['boundry']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        if opt.segmentation:
            self.netSeg = networks.define_seg(opt.input_nc_seg, opt.output_nc_seg,
                                              opt.which_model_netSeg, opt.norm,opt.init_type, not opt.no_dropout,
                                              self.gpu_ids, uncertainty=opt.uncertainty,mode=opt.mode_seg)
            

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netE = networks.define_D(opt.output_nc * 4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
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
            elif opt.segmentation and not opt.separate_seg:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(),self.netSeg.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_G)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            # self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
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

        if self.opt.continue_train:
            #  self.model_names = ['G', 'F', 'D', 'E']
            self.load_network_seg(self.netG, 'G', opt.which_epoch)
            self.load_network_seg(self.netF, 'F', opt.which_epoch)
            self.load_network_seg(self.netD, 'D', opt.which_epoch)
            self.load_network_seg(self.netE, 'E', opt.which_epoch)
            self.load_network_seg(self.netSeg, 'Seg', opt.which_epoch)
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



            self.load_network_seg(self.netG, 'G', opt.which_epoch)
            self.load_network_seg(self.netSeg, 'Seg', opt.which_epoch)
            self.mse_A = 0
            self.mse_B = 0
            self.mse_fake_A = 0
            self.mse_fake_B = 0
            # self.seg_fake = []
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
            self.image_array = np.zeros((30, 136, 156), dtype=np.float32)

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
        if self.number_of_images % 41 ==0:
            self.image_array = np.zeros((30, 136, 156), dtype=np.float32)

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
            # self.seg_fake = []
            self.dice_seg_real = []

    def data_dependent_initialize(self, data, data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data, data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:

            self.compute_G_loss().backward()
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()
            if self.opt.segmentation and self.opt.separate_seg and self.opt.mode_seg=='2d':
                self.loss_seg.backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        if self.opt.segmentation:
            self.netSeg.train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)

        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        if self.opt.segmentation and self.opt.separate_seg and self.opt.mode_seg=='2d'or (self.opt.mode_seg=='3d'and self.data_number%41 ==40 and self.opt.separate_seg):
            self.optimizer_seg.zero_grad()
        self.loss_G = self.compute_G_loss()
        # self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
        if self.opt.segmentation and self.opt.separate_seg and self.opt.mode_seg=='2d'or (self.opt.mode_seg=='3d'and self.data_number%41 ==40 and self.opt.separate_seg):
            self.loss_seg.backward()
            self.optimizer_seg.step()

    def set_input(self, input, input2=None,d=0):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths_A = input['A_paths']
        self.image_paths_B = input['B_paths']
        self.image_paths_seg = input['Seg_paths']
        self.dist_map_label = input['dist_map']

        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.segmentation:
            if self.opt.mode_seg == '2d':
                from torch.autograd import Variable
                self.real_Seg= input['Seg_one'].to(self.device)
                self.real_Seg = Variable(self.real_Seg.long())
                input_Seg = input['Seg']
                self.input_Seg.resize_(input_Seg.size()).copy_(input_Seg)
                self.input_Seg = Variable(self.input_Seg)

            elif self.opt.mode_seg =='3d':
                self.data_number = d
                seg_2D = input['Seg_one']

                if d%41 < self.opt.Depth:
                    # print('in setting 3D seg in input ', d%41)
                    self.input_Seg_one[0, d%41, :, :] = seg_2D
                    self.real_B_3d[:, :, d%41, :, :] = self.real_B

        #********

        # input_B = input['B']
        # self.input_B.resize_(input_B.size()).copy_(input_B)
        # ********
            
            

    def forward(self):
        # print('opt.nce_idt',self.opt.nce_idt)

        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1 / (i + 1) for i in range(T - 1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times).float().cuda()
        self.times = times
        bs = self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[bs]).cuda()).long()
        self.time_idx = time_idx
        self.timestep = times[time_idx]
        if self.opt.phase == 'train':
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.time_idx[0].int().item() + 1):
                    # print('t : ',t)
    
                    if t > 0:
                        delta = times[t] - times[t - 1]
                        denom = times[-1] - times[t - 1]
                        inter = (delta / denom).reshape(-1, 1, 1, 1)
                        scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                    Xt = self.real_A if (t == 0) else (1 - inter) * Xt + inter * Xt_1.detach() + (
                                scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time = times[time_idx]
                    z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(self.real_A.device)
                    # print('real_A ',Xt.shape,z.shape)
                    Xt_1 = self.netG(Xt, time_idx, z)
    
                    Xt2 = self.real_A2 if (t == 0) else (1 - inter) * Xt2 + inter * Xt_12.detach() + (
                                scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time = times[time_idx]
                    z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(self.real_A.device)
                    # print('real_A2 ', Xt2.shape, z.shape)
                    Xt_12 = self.netG(Xt2, time_idx, z)
                    if self.opt.Generation_step_seg:
                        seg = self.netSeg(Xt_12)


    
                    if self.opt.nce_idt:
                        XtB = self.real_B if (t == 0) else (1 - inter) * XtB + inter * Xt_1B.detach() + (
                                    scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                        time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                        time = times[time_idx]
                        z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(self.real_A.device)
                        # print('real_idt ',XtB.shape, z.shape)
                        Xt_1B = self.netG(XtB, time_idx, z)
                        if self.opt.Generation_step_seg:
                            seg = self.netSeg(Xt_1B)
                if self.opt.nce_idt:
                    self.XtB = XtB.detach()
                self.real_A_noisy = Xt.detach()
                self.real_A_noisy2 = Xt2.detach()

            z_in = torch.randn(size=[bs, 4 * self.opt.ngf]).to(self.real_A.device)
            # if self.opt.nce_idt and self.opt.isTrain :
            #     print('bssss',bs)
                # z_in = torch.randn(size=[2 * bs, 4 * self.opt.ngf]).to(self.real_A.device)
                # self.time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[2*bs]).cuda()).long()#(t * torch.ones(size=[2*self.real_A.shape[0]]).to(self.real_A.device)).long()
    
            z_in2 = torch.randn(size=[bs, 4 * self.opt.ngf]).to(self.real_A.device)
            """Run forward pass"""
            self.real = torch.cat((self.real_A, self.real_B),
                                  dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
    
            self.realt = torch.cat((self.real_A_noisy, self.XtB),
                                   dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy
    
            if self.opt.flip_equivariance:
                self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
                if self.flipped_for_equivariance:
                    self.real = torch.flip(self.real, [3])
                    self.realt = torch.flip(self.realt, [3])
    
            # print('real_t ', self.realt.shape, z_in.shape)
            # print(self.realt.shape, self.time_idx.shape, z_in.shape)
            self.fake = self.netG(self.realt, self.time_idx, z_in)
            # print('real_A_noisy_2 ',self.real_A_noisy2.shape,  z_in2.shape)
            self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2)
            self.fake_B = self.fake[:self.real_A.size(0)]
            if self.opt.nce_idt:
                self.idt_B = self.fake[self.real_A.size(0):]
            if self.opt.segmentation:

                if self.opt.separate_seg:
                    fake = self.fake_B.detach()
                    if self.opt.mode_seg == '2d':
                        if self.opt.add_quantum_noise:
                            fake = self.add_poisson_noise(fake)
                        self.fake_seg = self.netSeg(fake)
                    elif self.opt.mode_seg == '3d':
                        d = self.data_number
                        if d % 41 < self.opt.Depth:
                            # print(' in setting 3d fake B')
                            self.fake_B_3d[:, :, d % 41, :, :] = fake
                        if self.data_number % 41 == 40 and self.data_number:
                            # print(' in segmentaion 3d')
                            self.fake_seg = self.netSeg(self.fake_B_3d)

                else:
                    if self.opt.mode_seg == '2d':
                        if self.opt.add_quantum_noise:
                            self.fake_B = self.add_poisson_noise(self.fake_B)
                        self.fake_seg = self.netSeg(self.fake_B)
                        if self.opt.mri_ct_seg:
                            self.seg_real_A = self.netSeg(self.real_A)
                    if self.opt.mode_seg == '3d':
                        d = self.data_number
                        if d % 41 < self.opt.Depth:
                            # print(' in setting 3d fake B')
                            self.fake_B_3d[:, :, d % 41, :, :] = self.fake_B
                        if self.data_number % 41 == 40 and self.data_number:
                            # print(' in segmentaion 3d')
                            self.fake_seg = self.netSeg(self.fake_B_3d)


                # setattr(self, "seg_fake", self.seg_fake_B)


                

        if self.opt.phase == 'test':
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1 / (i + 1) for i in range(T - 1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1), times])
            times = torch.tensor(times).float().cuda()
            self.times = times
            bs = self.real_A.size(0)
            time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[bs]).cuda()).long()
            self.time_idx = time_idx
            self.timestep = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                # self.netSeg.eval()
                for t in range(self.opt.num_timesteps):

                    if t > 0:
                        delta = times[t] - times[t - 1]
                        denom = times[-1] - times[t - 1]
                        inter = (delta / denom).reshape(-1, 1, 1, 1)
                        scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                    Xt = self.real_A if (t == 0) else (1 - inter) * Xt + inter * Xt_1.detach() + (
                                scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time = times[time_idx]
                    z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(self.real_A.device)
                    Xt_1 = self.netG(Xt, time_idx, z)
                    setattr(self, "uncertainty_fake_CT_" + str(t + 1), Xt_1)
                    setattr(self, "fake_" + str(t + 1), Xt_1)
                    self.fake_B = Xt_1
                    if self.opt.mode_seg == '2d':
                        fake_seg = self.netSeg.forward(self.fake_B)
                        real_seg = self.netSeg.forward(self.real_B)
                        setattr(self, "uncertainty_fake_CT_seg" + str(t + 1), fake_seg)
                        setattr(self, "uncertainty_real_CT_seg" + str(t + 1), real_seg)

                if self.opt.mode_seg == '2d':
                    self.fake_seg = self.netSeg.forward(self.fake_B)
                    self.real_seg = self.netSeg.forward(self.real_B)
                # elif self.opt.mode_seg == '3d':
                #     d = self.data_number
                #     if d % 41 < self.opt.Depth:
                #         # print(' in setting 3d fake B')
                #         self.fake_B_3d[:, :, d % 41, :, :] = self.fake_B
                #
                #     if self.data_number % 41 == 40 and self.data_number:
                #         # print(' in segmentaion 3d')
                #         self.fake_seg = self.netSeg.forward(self.fake_B_3d)
                #         self.real_seg = self.netSeg.forward(self.real_B_3d)


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        bs = self.real_A.size(0)

        fake = self.fake_B.detach()
        std = torch.rand(size=[1]).item() * self.opt.std

        pred_fake = self.netD(fake, self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B, self.time_idx)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_E_loss(self):

        bs = self.real_A.size(0)

        """Calculate GAN loss for the discriminator"""

        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() + temp + temp ** 2

        return self.loss_E

    def compute_G_loss(self):
        bs = self.real_A.size(0)
        tau = self.opt.tau

        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        std = torch.rand(size=[1]).item() * self.opt.std

        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0
        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)

            bs = self.opt.batchSize

            ET_XY = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(
                self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            self.loss_SB = -(self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps * self.opt.tau * ET_XY
            self.loss_SB += self.opt.tau * torch.mean((self.real_A_noisy - self.fake_B) ** 2)
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + self.loss_SB + loss_NCE_both

        if self.opt.segmentation:
            if self.opt.mode_seg == '2d':
                arr = np.array(self.opt.crossentropy_weight)
                weight = torch.from_numpy(arr).cuda().float()
                # print(self.seg_fake_B.shape, self.real_Seg.shape, self.seg_fake_B.type(), self.real_Seg.type())
                self.loss_seg = networks.CrossEntropy2d(self.fake_seg, self.real_Seg, weight=weight)
                if self.opt.mri_ct_seg:
                    self.loss_seg += networks.CrossEntropy2d(self.seg_real_A, self.real_Seg, weight=weight)
                    self.loss_seg = self.loss_seg / 2.0
                if self.opt.boundry_loss:
                    self.loss_boundry = self.surface_loss(F.softmax(self.fake_seg, dim=1),
                                                          self.dist_map_label.cuda()) / 10.0
                    b_weight = self.opt.boundry_loss_weight
                    # print('CE loss',self.loss_seg, self.loss_boundry)
                    self.loss_seg = self.loss_seg + self.loss_boundry
                    # print('***', self.loss_seg)
                if not self.opt.separate_seg:
                    self.loss_G += (self.loss_seg * self.opt.seg_weight)

            elif self.opt.mode_seg == '3d':
                if self.data_number % 41 == 40 and self.data_number:
                    self.real_Seg = Variable(self.input_Seg_one.long())
                    # print('in 3D cross entropy ',self.data_number % 41 ,self.fake_seg.shape, self.real_Seg.shape)
                    self.loss_seg = self.criterion_seg(self.fake_seg, self.real_Seg)

                    if not self.opt.separate_seg:
                        self.loss_G += (self.loss_seg * self.opt.seg_weight)




        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z = torch.randn(size=[self.real_A.size(0), 4 * self.opt.ngf]).to(self.real_A.device)
        feat_q = self.netG(tgt, self.time_idx * 0, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.time_idx * 0, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        real_A = util.tensor2im(self.real_A.data)
        setattr(self, "visual_real_A", real_A)

        if self.opt.segmentation and self.opt.phase == 'test':

            # print('real seg 1 ')
            self.visual_seg_real = util.tensor2seg(self.seg_real.data[:,1,:,:])

        if self.opt.phase == 'test':
            for t in range(self.opt.num_timesteps):
                # print('get image in time step t: ',t )

                fake_B_t = util.tensor2im(getattr(self, "fake_" + str(t + 1)))
                setattr(self, "visual_fake_" + str(t + 1), fake_B_t)
                if self.opt.segmentation:
                    seg_fake = getattr(self, "seg_fake_" + str(t + 1))
                    visual_seg = util.tensor2seg(torch.max(seg_fake, dim=1, keepdim=True)[1])
                    setattr(self, "visual_seg_fake_" + str(t + 1), visual_seg)
        else:
            seg_fake = getattr(self, "seg_fake")
            visual_seg_fake = util.tensor2seg(torch.max(seg_fake, dim=1, keepdim=True)[1])
            setattr(self, "visual_seg_fake", visual_seg_fake)

            visual_fake_B = util.tensor2im(self.fake_B.data)
            setattr(self, "visual_fake_B", visual_fake_B)




    # def get_current_visuals(self):
    #     return_images = []
    #     return_images += [('real_A', getattr(self,'visual_real_A'))]
    #     if self.opt.segmentation and self.opt.phase == 'test':
    #         return_images += [('seg_real', getattr(self, 'visual_seg_real'))]
    #         for t in range(self.opt.num_timesteps):
    #             return_images += [(f'seg_fake_{t}', getattr(self, 'visual_seg_fake_' + str(t + 1)))]
    #             return_images += [(f'fake_B_{t}', getattr(self, 'visual_fake_' + str(t + 1)))]
    #     elif self.opt.segmentation:
    #         return_images += [(f'seg_fake', getattr(self, 'visual_seg_fake'))]
    #         return_images += [(f'fake_B', getattr(self, 'visual_fake_B' ))]
    #
    #     # for t in range(self.opt.num_timesteps):
    #
    #     return OrderedDict(return_images)


    def get_current_visuals(self):
        if self.opt.phase != 'test':
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)

            real_B = util.tensor2im(self.real_B.data)
            input_Seg = util.tensor2seg(self.input_Seg.data)

            # fake_seg = util.tensor2seg(torch.max(F.softmax(self.fake_seg.data, dim=1), dim=1, keepdim=True)[1])
            # real_seg = util.tensor2seg(torch.max(F.softmax(self.real_seg.data, dim=1), dim=1, keepdim=True)[1])
            # print('fake ****')
            fake_seg = util.tensor2seg(F.softmax(self.fake_seg.data, dim=1)[:,1,:,:])
            # print('real ****')
            # real_seg = util.tensor2seg(F.softmax(self.real_seg.data, dim=1)[:,1,:,:])#>0.005
            # print('fake ****')
            # fake_seg = util.tensor2seg( torch.sigmoid(self.fake_seg.data[:, 1, :, :]))
            # print('real ****')
            # real_seg = util.tensor2seg( torch.sigmoid(self.real_seg.data[:, 1, :, :]))  # >0.005


            return OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                                ('real_B', real_B),
                                ('fake_seg', fake_seg), ('input_seg', input_Seg)])#,('real_seg', real_seg)
        elif self.opt.phase == 'test':
            visuals = {}

            real_A= util.tensor2im(self.real_A.data)
            visuals['fake_B'] = [util.tensor2im(self.fake_B.data)]

            real_B = util.tensor2im(self.real_B.data)
            input_seg = util.tensor2seg(self.input_Seg.data)

            # print('fake ****')
            visuals['fake_seg'] = [util.tensor2seg(F.softmax(self.fake_seg.data, dim=1)[:, 1, :, :])]
            # print('real ****')
            visuals['seg_real'] = [util.tensor2seg(F.softmax(self.real_seg.data, dim=1)[:,1,:,:])]#>0.005
            # print('fake ****')

            # print('heatmap')
            heatmap = {key: util.tensor2seg(self.heatmap[key]) for key in self.heatmap.keys() if 'seg' in key}
            # print('uncertainty_map')
            uncertainty_map = {key: util.tensor2map(self.uncertainty_map[key]) for key in self.uncertainty_map.keys()}
            # print('confidence_map')
            confidence_map = {key: util.tensor2map(self.confidence_map[key]) for key in self.confidence_map.keys() if
                              'seg' in key}
            # print('entropy_map')
            entropy_map = {key: util.tensor2map(self.entropy_map[key]) for key in self.entropy_map.keys() if
                           'seg' in key}
            # print('var_map')
            var_map = {key: util.tensor2map(self.var[key]) for key in self.var.keys() if 'seg' not in key}

            real_A = util.tensor2im(self.real_A.data)
            real_B = util.tensor2im(self.real_B.data)
            # print('seg**')
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
                # print('*&*&*&', m.__class__.__name__)
                if m.__class__.__name__.startswith('Dropout'):
                    # print('******', m.__class__.__name__)
                    m.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        # print('i am in test')
        if not self.opt.MC_uncertainty:
            # self.enable_dropout()
            self.forward()
            # self.fake_B = self.netG.forward(self.real_A)
            # self.fake_seg = self.netSeg.forward(self.fake_B)
            # self.real_seg = self.netSeg.forward(self.real_B)


            # with torch.no_grad():
            #     self.enable_dropout()
            #     self.forward()
            #     self.compute_visuals()
        elif self.opt.MC_uncertainty :
            # print('number_of_images in model', self.number_of_images)
            if self.opt.mode_seg =='2d':
                self.create_uncertainty()
            elif self.opt.mode_seg =='3d':
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
                # print("Length of MC_uncertainty_outputs['fake_seg']:", len(self.MC_uncertainty_outputs['fake_seg']))  # Add this line

                self.MC_uncertainty_outputs['seg_real'].append(real_seg.detach().cpu())
                # print("Length of MC_uncertainty_outputs['seg_real']:", len(self.MC_uncertainty_outputs['seg_real']))  # Add this line

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
        # self.enable_dropout()

        if not self.opt.diff_uncertainty:
            print('i am here')
            for i in range(0, self.opt.num_samples_uncertainty):
                self.forward()
                fake_seg = F.softmax(self.fake_seg.data, dim=1)
                real_seg = F.softmax(self.real_seg.data, dim=1)
                fake_B = self.fake_B
                self.MC_uncertainty_outputs['fake_seg'].append(fake_seg.detach().cpu())

                self.MC_uncertainty_outputs['seg_real'].append(real_seg.detach().cpu())

                self.MC_uncertainty_outputs['fake_B'].append(fake_B.detach().cpu())
        else:
            self.forward()
            for i in range(0, self.opt.num_timesteps):
                fake_seg = F.softmax(getattr(self, "uncertainty_fake_CT_seg" + str(i + 1)), dim=1)
                real_seg = F.softmax(getattr(self, "uncertainty_real_CT_seg" + str(i + 1)), dim=1)
                fake_B = getattr(self, "uncertainty_fake_CT_" + str(i + 1))
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
                uncertainty_map[key] = var[key].max(dim=1)[0].sqrt()
                confidence_map[key] = mean[key].max(dim=1)[0]
                entropy_map[key] = self.compute_entropy(mean[key])
                if key =='seg_real' :

                    if self.number_of_images < 35:
                        self.cal_manual_dice(outputs[key])
                        self.cal_manual_iou(outputs[key])
                    else:
                        self.tn1, self.fp1, self.fn1, self.tp1 = [65536, 0, 0, 0]
                        self.tn2, self.fp2, self.fn2, self.tp2 = [65536, 0, 0, 0]
                        self.mean_pure_dice = self.mean_dice_gamma_1 = self.mean_dice_gamma_2 = torch.tensor(1.0, dtype=torch.float32).item()
                        self.one_pure_dice = self.one_dice_gamma_1 = self.one_dice_gamma_2 = torch.tensor(1.0, dtype=torch.float32).item()
                        self.mean_pure_iou = self.mean_iou_gamma_1 = self.mean_iou_gamma_2 = torch.tensor(1.0, dtype=torch.float32)
                        self.one_pure_iou = self.one_iou_gamma_1 = self.one_iou_gamma_2 = torch.tensor(1.0, dtype=torch.float32)




            else:
                stacked_output = torch.stack(outputs[key])
                mean[key] = stacked_output.mean(dim=0)
                var[key] = stacked_output.var(dim=0)
                uncertainty_map[key] = var[key].sqrt()[0]

        return mean, var, heatmap, uncertainty_map, confidence_map, entropy_map

    def cal_manual_dice(self, segmentation):
        # print('in cal manual dice ', np.shape(segmentation))
        mean_seg = torch.stack(segmentation).mean(dim=0)
        # print('in cal manual dice after mean', np.shape(mean_seg))

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

        # self.visual_gamma1 = torch.argmax(seg_gamma_1, dim=1)
        # self.visual_gamma2 = torch.argmax(seg_gamma_2, dim=1)

        self.processed_dice3d(y_pred1, y_pred2)
        return pure_dice, dice_gamma_1, dice_gamma_2
    def processed_dice3d(self, y_pred1, y_pred2):
        y_true = self.input_Seg[:, 1, :, :].view(-1).detach().cpu().numpy()
        y_pred1 = y_pred1.view(-1).detach().cpu().numpy()
        y_pred2 = y_pred2.view(-1).detach().cpu().numpy()

        self.tn1, self.fp1, self.fn1, self.tp1 = confusion_matrix(y_true, y_pred1, labels=[0, 1]).ravel()
        self.tn2, self.fp2, self.fn2, self.tp2 = confusion_matrix(y_true, y_pred2, labels=[0, 1]).ravel()


    # def preprocessed_gamma(self, seg):
    #     pure_dice = self.dice_coef(seg, self.input_Seg, smooth=1)
    #     seg_gamma_1 = self.apply_gamma(seg, 0.1)
    #     dice_gamma_1 = self.dice_coef(seg_gamma_1, self.input_Seg, smooth=1)
    #     seg_gamma_2 = self.apply_gamma(seg, 0.2)
    #     dice_gamma_2 = self.dice_coef(seg_gamma_2, self.input_Seg, smooth=1)
    #     return pure_dice, dice_gamma_1, dice_gamma_2

    def crop_images(self, seg):
        # print('****&*&*',np.shape(seg))
        height, width = seg.shape[2:]  # In case of color image, ignore the color channel
        left = 50
        right = width - 50
        top = 60
        bottom = height - 60
        cropped_image = seg[0, 1, top:bottom, left:right]
        # print('****&&&@@@*&*',np.shape(cropped_image))
        return cropped_image

    def pad_image(self, seg, cropped_image):
        # print('Original image shape:', seg.shape)
        # print('Cropped image shape:', cropped_image.shape)
        # Assuming seg is a 4D tensor with shape [N, C, H, W]
        height, width = seg.shape[2:]  # Get height and width

        # Define crop boundaries
        left, right = 50, width - 50
        top, bottom = 60, height - 60

        # Padded image (zero-filled)
        padded_image = torch.zeros_like(seg)
        padded_image[:, 1, top:bottom, left:right] = cropped_image
        # print('Padded image shape:', padded_image.shape)

        return padded_image

    def apply_gamma(self, seg, gamma):
        from skimage.filters import threshold_otsu
        from skimage import exposure
        # print('apply gamma ',np.shape(seg))
        cropped_image = self.crop_images(seg)
        # print('apply gamma after crop ',np.shape(cropped_image))

        cropped_image = self.pad_image(seg, cropped_image)
        # print('apply gamma after pad ',np.shape(cropped_image))

        if isinstance(cropped_image, torch.Tensor):
            cropped_image = cropped_image.detach().cpu().numpy()  # Convert to numpy
            cropped_image = cropped_image.astype(np.float32)

        cropped_image = exposure.adjust_gamma(cropped_image, gamma)
        # print('apply gamma after gamma ',np.shape(cropped_image))

        non_zero_cropped_image = cropped_image[cropped_image > 0]
        threshold_value = 0
        if len(non_zero_cropped_image) > 0:  # Check if there are non-zero values
            threshold_value = threshold_otsu(non_zero_cropped_image)
        threshold_value = threshold_value - 0.15
        binary_image = cropped_image > threshold_value
        # print('apply gamma after thresh ',np.shape(binary_image))

        binary_image = torch.from_numpy(binary_image)
        binary_image = binary_image.int()
        # print('apply gamma after apply gamma ',np.shape(binary_image))

        return binary_image

    # def compute_output(self, outputs):
    #     mean = {key: [] for key in outputs.keys()}
    #     var = {key: [] for key in outputs.keys()}
    #     heatmap = {key: [] for key in outputs.keys()}
    #     uncertainty_map = {key: [] for key in outputs.keys()}
    #     confidence_map = {key: [] for key in outputs.keys()}
    #     entropy_map = {key: [] for key in outputs.keys()}
    #     time = self.opt.num_timesteps - 1
    #     # mean['seg_real'], var['seg_real'] = self.probability(outputs['seg_real'])
    #     mean_real_seg, var_real_seg = self.probability(outputs['seg_real'])
    #
    #     heatmap['seg_real'].append({'t': time, 'value': torch.argmax(mean_real_seg, dim=1)})
    #     uncertainty_map['seg_real'].append({'t': time, 'value': var_real_seg.max(dim=1)[0].sqrt()})
    #     confidence_map['seg_real'].append({'t': time, 'value': mean_real_seg.max(dim=1)[0]})
    #     entropy_map['seg_real'].append({'t': time, 'value': self.compute_entropy(mean_real_seg)})
    #     mean['seg_real'].append({'t': time, 'value': mean_real_seg})
    #     var['seg_real'].append({'t': time, 'value': var_real_seg})
    #
    #     for t in range(self.opt.num_timesteps):
    #
    #         fake_B = [sample[t] for sample in outputs['fake_B']]
    #         mean_fake_B, var_fake_B = self.probability(fake_B)
    #         fake_seg = [sample[t] for sample in outputs['fake_seg']]
    #         mean_fake_seg, var_fake_seg = self.probability(fake_seg)
    #
    #         time = t
    #         uncertainty_map['fake_B'].append({'t': time, 'value': var_fake_B.sqrt()[0]})
    #         heatmap['fake_seg'].append({'t': time, 'value': torch.argmax(mean_fake_seg, dim=1)})
    #         uncertainty_map['fake_seg'].append({'t': time, 'value': var_fake_seg.max(dim=1)[0].sqrt()})
    #         confidence_map['fake_seg'].append({'t': time, 'value': mean_fake_seg.max(dim=1)[0]})
    #         entropy_map['fake_seg'].append({'t': time, 'value': self.compute_entropy(mean_fake_seg)})
    #
    #         mean['fake_seg'].append({'t': time, 'value': mean_fake_seg})
    #         var['fake_seg'].append({'t': time, 'value': var_fake_seg})
    #         mean['fake_B'].append({'t': time, 'value': mean_fake_B})
    #         var['fake_B'].append({'t': time, 'value': var_fake_B})
    #     return mean, var, heatmap, uncertainty_map, confidence_map, entropy

    def probability(self,img):
        stacked_output = torch.stack(img)
        mean = stacked_output.mean(dim=0)
        var = stacked_output.var(dim=0)
        # for i, item in enumerate(img):
        # print(f"Shape of item : {np.shape(img)}")
        return mean, var
    # def set_coef(self):
    #     self.var_fake_B_2D = self.avg_uncertainty(self.uncertainty_map['fake_B'])
    #     self.var_seg_real_2D = self.avg_uncertainty(self.uncertainty_map['seg_real'])
    #     self.var_seg_fake_2D = self.avg_uncertainty(self.uncertainty_map['fake_seg'])
    #     self.ssim_value_B_2D = ssim(self.mean['fake_B'].cuda(), self.real_B, data_range=1.0,
    #                                 size_average=True).item()
    #     self.mse_fake_B_2D = self.cal_mse(self.mean['fake_B'], self.real_B)
    #     self.seg_real_IOU_2D = self.get_IOU(self.heatmap['seg_real'], self.input_Seg)
    #     self.seg_fake_2D = self.dice_coef(self.heatmap['fake_seg'], self.input_Seg, smooth=1)
    #     self.seg_real_2D = self.dice_coef(self.heatmap['seg_real'], self.input_Seg, smooth=1)
    #     self.get_confusion_matrix(self.heatmap['seg_real'], self.input_Seg, 'real')
    #     self.recal_2D = self.cal_recal()
    #     self.specificity_2D = self.cal_specificity()
    #     self.percision_2D = self.cal_percision()
    #
    #     self.var_fake_B += self.var_fake_B_2D
    #
    #     self.var_seg_real += self.var_seg_real_2D
    #     self.var_seg_fake += self.var_seg_fake_2D
    #
    #     self.ssim_value_B += self.ssim_value_B_2D
    #     self.mse_fake_B += self.mse_fake_B_2D
    #
    #     self.seg_real_IOU += self.seg_real_IOU_2D
    #
    #     self.seg_fake.append(self.seg_fake_2D)
    #     self.dice_seg_real.append(self.seg_real_2D)
    #
    # #
    # def get_coef(self):
    #     # dice_seg_fake = np.mean(self.seg_fake)
    #     # std_dice_seg_fake = np.std(self.seg_fake)
    #     #
    #     # # print('self.seg_real',self.seg_real)
    #     # dice_seg_real = np.mean(self.dice_seg_real)
    #     # std_dice_seg_real = np.std(self.dice_seg_real)
    #
    #     results = \
    #         {
    #             'mean_pure_iou':self.mean_pure_iou,
    #             'mean_iou_gamma_1':self.mean_iou_gamma_1,
    #             'mean_iou_gamma_2':self.mean_iou_gamma_2,
    #             'one_pure_iou':self.one_pure_iou,
    #             'one_iou_gamma_1': self.one_iou_gamma_1,
    #             'one_iou_gamma_2': self.one_iou_gamma_2,
    #
    #             'mean_pure_dice': self.mean_pure_dice,
    #             'mean_dice_gamma_1':self.mean_dice_gamma_1,
    #             'mean_dice_gamma_2':self.mean_dice_gamma_2,
    #             'one_pure_dice':self.one_pure_dice,
    #             'one_dice_gamma_1':self.one_dice_gamma_1,
    #             'one_dice_gamma_2':self.one_dice_gamma_2,
    #
    #
    #             # 'mse_fake_B': self.mse_fake_B,
    #             'mse_fake_B_2D': self.mse_fake_B_2D,
    #             # 'seg_real_IOU_H': self.seg_real_IOU,
    #             'seg_real_IOU_2D_H': self.seg_real_IOU_2D,
    #             'dice_seg_fake_2D_H': self.seg_fake_2D,
    #             'dice_seg_real_2D_H': self.seg_real_2D,
    #             # 'mean_dice_seg_fake': dice_seg_fake,
    #             # 'mean_dice_seg_real': dice_seg_real,
    #             # 'std_dice_seg_fake': std_dice_seg_fake,
    #             # 'std_dice_seg_real': std_dice_seg_real,
    #             # 'ssim_value_B': self.ssim_value_B,
    #             'ssim_value_B_2D': self.ssim_value_B_2D,
    #
    #             # 'var_seg_fake': self.var_seg_fake,
    #             'avg_var_seg_fake_2D': self.var_seg_fake_2D,
    #
    #             # 'var_seg_real': self.var_seg_real,
    #             'avg_var_seg_real_2D': self.var_seg_real_2D,
    #
    #             # 'var_fake_B': self.var_fake_B,
    #             'avg_var_fake_B_2D': self.var_fake_B_2D,
    #
    #             'percision_2D': self.percision_2D,
    #             'specificity_2D': self.specificity_2D,
    #             'recal_2D': self.recal_2D,
    #             # 'tp': self.tp,
    #             # 'tn': self.tn,
    #             # 'fp': self.fp,
    #             # 'fn': self.fn,
    #             'dice': self.dice,
    #             'iou': self.iou,
    #             'recall': self.recal,
    #             'specificity': self.specificity,
    #             'percision': self.percision,
    #             'CT_name': self.image_paths_B[0].split('/')[-1],
    #             'mri_name': self.image_paths_A[0].split('/')[-1],
    #             'Seg': self.image_paths_seg[0].split('/')[-1],
    #         }
    #     return results


    # def set_coef(self):
    #
    #     self.var_seg_real_2D = self.avg_uncertainty(self.uncertainty_map['seg_real'][0]['value'])
    #     self.seg_real_IOU_2D = self.get_IOU(self.heatmap['seg_real'][0]['value'], self.input_Seg)
    #     self.seg_real_2D = self.dice_coef(self.heatmap['seg_real'][0]['value'], self.input_Seg, smooth=1)
    #     self.get_confusion_matrix(self.heatmap['seg_real'][0]['value'], self.input_Seg, 'real')
    #     self.recal_2D = self.cal_recal()
    #     self.specificity_2D = self.cal_specificity()
    #     self.percision_2D = self.cal_percision()
    #     self.var_seg_real += self.var_seg_real_2D
    #     self.seg_real_IOU += self.seg_real_IOU_2D
    #     self.dice_seg_real.append(self.seg_real_2D)
    #
    #     self.results['seg_real_IOU']= self.seg_real_IOU.item()
    #     self.results['seg_real_IOU_2D']= self.seg_real_IOU_2D.item()
    #     self.results['dice_seg_real_2D']= self.seg_real_2D
    #     self.results['var_seg_real'] = self.var_seg_real
    #     self.results['var_seg_real_2D'] = self.var_seg_real_2D
    #
    #     self.results['percision_2D'] = self.percision_2D
    #     self.results['specificity_2D'] = self.specificity_2D
    #     self.results['recal_2D'] = self.recal_2D
    #
    #     self.results['tp'] = self.tp
    #     self.results['tn'] = self.tn
    #     self.results['fp'] = self.fp
    #     self.results['fn'] = self.fn
    #
    #     self.results['dice'] = self.dice
    #     self.results['iou'] = self.iou
    #     self.results['recall'] = self.recal
    #     self.results['specificity'] = self.specificity
    #     self.results['percision'] = self.percision
    #
    #
    #     self.var_fake_B = {t: 0 for t in range(self.opt.num_timesteps)}
    #     self.var_seg_fake = {t: 0 for t in range(self.opt.num_timesteps)}
    #     self.ssim_value_B = {t: 0 for t in range(self.opt.num_timesteps)}
    #     self.mse_B = {t: 0 for t in range(self.opt.num_timesteps)}
    #     self.mse_fake_B = {t: 0 for t in range(self.opt.num_timesteps)}
    #     self.seg_fake = {t: [] for t in range(self.opt.num_timesteps)}
    #
    #     for t in range(self.opt.num_timesteps):
    #
    #         self.var_fake_B_2D = self.avg_uncertainty(self.uncertainty_map['fake_B'][t]['value'])
    #         self.ssim_value_B_2D = ssim(self.mean['fake_B'][t]['value'].cuda(), self.real_B, data_range=1.0, size_average=True).item()
    #         self.mse_fake_B_2D = self.cal_mse(self.mean['fake_B'][t]['value'], self.real_B)
    #
    #         self.var_seg_fake_2D = self.avg_uncertainty(self.uncertainty_map['fake_seg'][t]['value'])
    #         self.seg_fake_2D = self.dice_coef(self.heatmap['fake_seg'][t]['value'], self.input_Seg, smooth=1)
    #
    #
    #         self.var_fake_B[t] += self.var_fake_B_2D
    #         self.ssim_value_B[t] += self.ssim_value_B_2D
    #         self.mse_fake_B[t] += self.mse_fake_B_2D
    #
    #         self.seg_fake[t].append(self.seg_fake_2D)
    #         self.var_seg_fake[t] += self.var_seg_fake_2D
    #
    #         self.results[f'mse_fake_B_{t}'] =  self.mse_fake_B[t]
    #         self.results[f'mse_fake_B_2D_{t}'] = self.mse_fake_B_2D
    #         self.results[f'dice_seg_fake_2D_{t}'] = self.seg_fake_2D
    #         self.results[f'ssim_value_B_{t}'] = self.ssim_value_B[t]
    #         self.results[f'ssim_value_B_2D_{t}'] = self.ssim_value_B_2D
    #
    #         self.results[f'var_seg_fake_{t}'] = self.var_seg_fake[t]
    #         self.results[f'var_seg_fake_2D_{t}'] =  self.var_seg_fake_2D
    #         self.results[f'var_fake_B_{t}'] =  self.var_fake_B[t]
    #         self.results[f'var_fake_B_2D_{t}'] = self.var_fake_B_2D
    #
    #
    #
    #
    #
    #
    # def get_coef(self):
    #
    #
    #     for t in range(self.opt.num_timesteps):
    #         dice_seg_fake = np.mean(self.seg_fake[t])
    #         std_dice_seg_fake = np.std(self.seg_fake[t])
    #         self.results[f'mean_dice_seg_fake_{t}'] = dice_seg_fake
    #         self.results[f'std_dice_seg_fake_{t}'] = std_dice_seg_fake
    #
    #
    #     dice_seg_real = np.mean(self.dice_seg_real)
    #     std_dice_seg_real = np.std(self.dice_seg_real)
    #     self.results['mean_dice_seg_real'] = dice_seg_real
    #     self.results['std_dice_seg_real'] = std_dice_seg_real
    #
    #
    #
    #
    #     self.results['CT_name'] = self.image_paths_B[0].split('/')[-1]
    #     self.results['mri_name'] =  self.image_paths_A[0].split('/')[-1]
    #     self.results['Seg'] = self.image_paths_seg[0].split('/')[-1]
    #
    #     return self.results

    def calculate_ssim(self,image1, image2):
        import torch
        import torchmetrics
        import torchvision.transforms.functional as F

        sigma = 2
        kernel_size = 7

        # Applying Gaussian blur
        image1 = F.gaussian_blur(image1, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        image2 = F.gaussian_blur(image2, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        # Initialize SSIM metric
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(kernel_size=11, data_range=1.0,
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

        # self.var_fake_B += self.var_fake_B_2D
        #
        # self.var_seg_real += self.var_seg_real_2D
        # self.var_seg_fake += self.var_seg_fake_2D
        #
        # self.ssim_value_B += self.ssim_value_B_2D
        # self.mse_fake_B += self.mse_fake_B_2D
        #
        # self.seg_real_IOU += self.seg_real_IOU_2D
        #
        # self.seg_fake.append(self.seg_fake_2D)
        # self.dice_seg_real.append(self.seg_real_2D)

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
                'new_ssim':self.new_ssim,

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
        # print(' in dice coef', np.shape(ground_truth), np.shape(predicted))
        intersection = torch.sum(predicted * ground_truth)
        dice = (2.0 * intersection + smooth) / (torch.sum(predicted) + torch.sum(ground_truth) + smooth)
        # print('dice ',dice)
        return dice.item()

    def get_IOU(self, predicted, ground_truth):
        ground_truth = ground_truth[:, 1, :, :]
        ground_truth = ground_truth.detach().cpu()
        # print(' in IOU', np.shape(ground_truth), np.shape(predicted))

        intersection = torch.sum(predicted * ground_truth)
        union = (predicted.bool() | ground_truth.bool()).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        # print('iou ', iou, intersection, union)
        return iou.mean()

    def cal_mse(self, input, target):
        target = target.detach().cpu()
        mse = torch.mean((input - target) ** 2)
        return mse.item()

    def get_confusion_matrix(self, y_pred, y_true, key):
        y_true = y_true[:, 1, :, :].detach().cpu().numpy()
        img1 = y_true  # .detach().cpu().numpy()

        img1 = img1.flatten()

        y_true = img1  # y_true.view(-1)#.detach().cpu().numpy()
        y_pred = y_pred.view(-1).detach().cpu().numpy()
        self.tn, self.fp, self.fn, self.tp = self.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        self.TP = self.TP + self.tp
        self.TN = self.TN + self.tn
        self.FP = self.FP + self.fp
        self.FN = self.FN + self.fn

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

        return similarity[0]



        



        