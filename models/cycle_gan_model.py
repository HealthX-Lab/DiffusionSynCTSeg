# import numpy as np
# import torch
# import os
# from collections import OrderedDict
# from torch.autograd import Variable
# import itertools
# import util.util as util
# from util.image_pool import ImagePool
# from .base_model import BaseModel
# from . import networks
# import sys
# from .early_stopping import EarlyStopping
#
#
# class CycleGANModel(BaseModel):
#     def name(self):
#         return 'CycleGANModel'
#
#     def initialize(self, opt):
#         BaseModel.initialize(self, opt)
#
#         nb = opt.batchSize
#         size = opt.fineSize
#         self.input_A = self.Tensor(nb, opt.input_nc, size, size)
#         self.input_B = self.Tensor(nb, opt.output_nc, size, size)
#
#         # load/define networks
#         # The naming conversion is different from those used in the paper
#         # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
#
#         self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
#                                         opt.ngf, opt.which_model_netG, opt.norm, self.gpu_ids)#, not opt.no_dropout
#         self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
#                                         opt.ngf, opt.which_model_netG, opt.norm, self.gpu_ids)#, not opt.no_dropout
#
#         if self.isTrain:
#             use_sigmoid = True
#             self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
#                                             opt.which_model_netD,
#                                             opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
#             self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
#                                             opt.which_model_netD,
#                                             opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
#         if not self.isTrain or opt.continue_train:
#             which_epoch = opt.which_epoch
#             self.load_network(self.netG_A, 'G_A', which_epoch)
#             self.load_network(self.netG_B, 'G_B', which_epoch)
#             if self.isTrain:
#                 self.load_network(self.netD_A, 'D_A', which_epoch)
#                 self.load_network(self.netD_B, 'D_B', which_epoch)
#
#         if self.isTrain:
#             self.old_lr = opt.lr
#             self.fake_A_pool = ImagePool(opt.pool_size)
#             self.fake_B_pool = ImagePool(opt.pool_size)
#             # define loss functions
#             if opt.Wasserstein_Lossy:
#                 self.criterionGAN = networks.WassersteinLoss()
#             else:
#                 self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor,
#                                                      target_real_label=1.0, target_fake_label=0.0)
#             self.criterionCycle = torch.nn.L1Loss()
#             self.criterionIdt = torch.nn.L1Loss()
#             # initialize optimizers
#             self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
#                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#
#         print('---------- Networks initialized -------------')
#         networks.print_network(self.netG_A)
#         networks.print_network(self.netG_B)
#         if self.isTrain:
#             networks.print_network(self.netD_A)
#             networks.print_network(self.netD_B)
#         print('-----------------------------------------------')
#
#         # early stopping
#         if opt.enable_early_stopping:
#             print('in early stopping  &&&&')
#             self.best_metric = {'G_A': float('inf'), 'G_B': float('inf'), 'D_A': float('inf'), 'D_B': float('inf')}
#             self.epochs_since_improvement = {'G_A': 0, 'G_B': 0, 'D_A': 0, 'D_B': 0}
#             self.disable_training = {'G_A': 0, 'G_B': 0, 'D_A': 0, 'D_B': 0}
#             self.stop_training = 0
#
#
#
#     def get_enable_training_model(self):
#         list_enable = []
#         for key, value in self.disable_training.items():
#             if value == 0:
#                 list_enable.append(key)
#         return list_enable
#
#     def set_input(self, input):
#         AtoB = self.opt.which_direction == 'AtoB'
#         input_A = input['A' if AtoB else 'B']
#         input_B = input['B' if AtoB else 'A']
#         self.input_A.resize_(input_A.size()).copy_(input_A)
#         self.input_B.resize_(input_B.size()).copy_(input_B)
#         self.image_paths = input['A_paths' if AtoB else 'B_paths']
#
#     def forward(self):
#         self.real_A = Variable(self.input_A)
#         self.real_B = Variable(self.input_B)
#
#     def test(self):
#         self.real_A = Variable(self.input_A)#, volatile=True
#         self.fake_B = self.netG_A.forward(self.real_A)
#         self.rec_A = self.netG_B.forward(self.fake_B)
#
#         self.real_B = Variable(self.input_B)#, volatile=True
#         self.fake_A = self.netG_B.forward(self.real_B)
#         self.rec_B = self.netG_A.forward(self.fake_A)
#
#         if self.opt.identity > 0:
#             # print('*********** in identity')
#             # G_A should be identity if real_B is fed.
#             self.idt_A = self.netG_A.forward(self.real_B)
#             self.idt_B = self.netG_B.forward(self.real_A)
#
#
#     # get image paths
#     def get_image_paths(self):
#         return self.image_paths
#
#     def gradient_penalty(self,discriminator, real_samples, fake_samples):
#         batch_size = real_samples.size(0)
#         # epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
#         epsilon = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
#         interpolates = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)
#         d_interpolates = discriminator(interpolates).cuda()
#
#         gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
#                                         grad_outputs=torch.ones(d_interpolates.size(),device='cuda'),
#                                         create_graph=True, retain_graph=True)[0]
#         gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
#         penalty = ((gradient_norm - 1) ** 2).mean()
#         return penalty
#
#     def backward_D_basic(self, netD, real, fake):
#         loss_D = 0
#         if not self.opt.Wasserstein_Lossy:
#             # Real
#             pred_real = netD.forward(real)
#             loss_D_real = self.criterionGAN(pred_real, True)
#             # Fake
#             pred_fake = netD.forward(fake.detach())
#             loss_D_fake = self.criterionGAN(pred_fake, False)
#             # Combined loss
#             loss_D = (loss_D_real + loss_D_fake) * 0.5
#
#         elif self.opt.Wasserstein_Lossy:
#             real_logits = netD.forward(real)
#             generated_logits = netD.forward(fake.detach())
#             loss_D = self.criterionGAN(real_logits, generated_logits, discriminator=True)
#             if self.opt.gradient_penalty:
#                 gradient_penalty_loss = self.gradient_penalty(netD, real, fake)
#                 loss_D = loss_D + self.opt.gradient_penalty_lambda * gradient_penalty_loss
#
#         loss_D.backward()
#         return loss_D
#
#     def backward_D_A(self):
#         fake_B = self.fake_B_pool.query(self.fake_B)
#         self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
#
#     def backward_D_B(self):
#         fake_A = self.fake_A_pool.query(self.fake_A)
#         self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
#
#     def backward_G(self):
#         # print('indentiryyyyy',self.opt.identity)
#         lambda_idt = self.opt.identity
#         lambda_A = self.opt.lambda_A
#         lambda_B = self.opt.lambda_B
#         # Identity loss
#         if lambda_idt > 0:
#             # print('*********** in identity')
#             # G_A should be identity if real_B is fed.
#             self.idt_A = self.netG_A.forward(self.real_B)
#             self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
#             # G_B should be identity if real_A is fed.
#             self.idt_B = self.netG_B.forward(self.real_A)
#             self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
#         else:
#             self.loss_idt_A = 0
#             self.loss_idt_B = 0
#
#         # GAN loss
#         # D_A(G_A(A))
#         self.fake_B = self.netG_A.forward(self.real_A)
#         pred_fake = self.netD_A.forward(self.fake_B)
#         if not self.opt.Wasserstein_Lossy:
#             self.loss_G_A = self.criterionGAN(pred_fake, True)
#         elif self.opt.Wasserstein_Lossy:
#             self.loss_G_A = self.criterionGAN(generated_logits=pred_fake, discriminator=False)
#         print('self.loss_G_A:  ', self.loss_G_A)
#         # D_B(G_B(B))
#         self.fake_A = self.netG_B.forward(self.real_B)
#         pred_fake = self.netD_B.forward(self.fake_A)
#         if not self.opt.Wasserstein_Lossy:
#             self.loss_G_B = self.criterionGAN(pred_fake, True)
#         elif self.opt.Wasserstein_Lossy:
#             self.loss_G_B = self.criterionGAN(generated_logits=pred_fake, discriminator=False)
#         print('self.loss_G_B:  ', self.loss_G_B)
#         # Forward cycle loss
#         self.rec_A = self.netG_B.forward(self.fake_B)
#         self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
#         # Backward cycle loss
#         self.rec_B = self.netG_A.forward(self.fake_A)
#         self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
#         # combined loss
#         self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
#         self.loss_G.backward()
#
#     def optimize_parameters(self,epoch=0):
#         if not self.opt.enable_early_stopping:
#             # forward
#             self.forward()
#             # G_A and G_B
#             self.optimizer_G.zero_grad()
#             self.backward_G()
#             self.optimizer_G.step()
#             # D_A
#             self.optimizer_D_A.zero_grad()
#             self.backward_D_A()
#             self.optimizer_D_A.step()
#             # D_B
#             self.optimizer_D_B.zero_grad()
#             self.backward_D_B()
#             self.optimizer_D_B.step()
#
#         elif self.opt.enable_early_stopping:
#             if len(self.get_enable_training_model()) == 0:
#                 self.stop_training = 1
#                 return
#             self.forward()
#             # G_A and G_B
#             self.optimizer_G.zero_grad()
#             self.backward_G()
#             self.optimizer_G.step()
#             if not self.disable_training['D_A']:
#                 self.optimizer_D_A.zero_grad()
#                 self.backward_D_A()
#                 self.optimizer_D_A.step()
#             if not self.disable_training['D_B']:
#                 # D_B
#
#                 self.optimizer_D_B.zero_grad()
#                 self.backward_D_B()
#                 self.optimizer_D_B.step()
#
#     def get_current_errors(self):
#         D_A = self.loss_D_A.item()
#         G_A = self.loss_G_A.item()
#         Cyc_A = self.loss_cycle_A.item()
#         D_B = self.loss_D_B.item()
#         G_B = self.loss_G_B.item()
#         Cyc_B = self.loss_cycle_B.item()
#         if self.opt.identity > 0.0:
#             idt_A = self.loss_idt_A.item()
#             idt_B = self.loss_idt_B.item()
#             return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
#                                 ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
#         else:
#             return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
#                                 ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])
#
#     def get_current_visuals(self):
#         real_A = util.tensor2im_real(self.real_A.data)
#         fake_B = util.tensor2im(self.fake_B.data)
#         rec_A = util.tensor2im(self.rec_A.data)
#         real_B = util.tensor2im_real(self.real_B.data)
#         fake_A = util.tensor2im(self.fake_A.data)
#         rec_B = util.tensor2im(self.rec_B.data)
#         if self.opt.identity > 0.0:
#             idt_A = util.tensor2im(self.idt_A.data)
#             idt_B = util.tensor2im(self.idt_B.data)
#             return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
#                                 ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
#         else:
#             return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
#                                 ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
#
#     def save(self, label):
#         self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
#         self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
#         self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
#         self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
#
#     def save_individual_model(self, model_name, label):
#         if model_name == 'G_A':
#             self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
#         elif model_name == 'D_A':
#             self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
#         elif model_name == 'G_B':
#             self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
#         elif model_name == 'D_B':
#             self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
#
#     def earlyStopping(self, epoch):
#         loss_G_A = 0
#         loss_G_B = 0
#         if self.opt.identity > 0.0:
#             loss_G_A = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A
#             loss_G_B = self.loss_G_B + self.loss_cycle_B + self.loss_idt_B
#         else:
#             loss_G_A = self.loss_G_A + self.loss_cycle_A
#             loss_G_B = self.loss_G_B + self.loss_cycle_B
#
#         print('%^%^ in early stopping')
#         if not self.disable_training['G_A']:
#             self.check_eraly_stopping('G_A', self.netG_A, loss_G_A, self.opt.min_delta_G, epoch)
#         if not self.disable_training['G_B']:
#             self.check_eraly_stopping('G_B', self.netG_B, loss_G_B, self.opt.min_delta_G, epoch)
#
#         if not self.disable_training['D_A']:
#             self.check_eraly_stopping('D_A', self.netD_A, self.loss_D_A, self.opt.min_delta_D, epoch)
#         if not self.disable_training['D_B']:
#             self.check_eraly_stopping('D_B', self.netD_B, self.loss_D_B, self.opt.min_delta_D, epoch)
#
#
#     def check_eraly_stopping(self,name,model,loss,delta,epoch):
#         print('in check early stopping ', name)
#         if self.best_metric[name] - loss >=  delta:
#             self.best_metric[name] = loss
#             self.epochs_since_improvement[name] = 0
#             # Save the best model
#             self.save_individual_model(name,f'best_{name}')
#         else:
#             print('I am in plus for ', name)
#             self.epochs_since_improvement[name] += 1
#
#         patience = 0
#         if 'D' in name:
#             if self.epochs_since_improvement[name] >= self.opt.patience_D and epoch<60:
#                 self.opt.patience_D = self.opt.patience_D + 5
#
#             patience = self.opt.patience_D
#         else:
#             if self.epochs_since_improvement[name] >= self.opt.patience_G and epoch<60:
#                 self.opt.patience_G = self.opt.patience_G + 5
#
#             patience = self.opt.patience_G
#
#         if self.epochs_since_improvement[name] >= patience:
#             print(f"Early stopping for {name}in epoch {epoch} patience {patience}!")
#             self.save_individual_model(name, f'best_{name}')
#             for param in model.parameters():
#                 param.requires_grad = False
#             self.disable_training[name] = 1
#
#     def update_learning_rate(self):
#         lrd = self.opt.lr / self.opt.niter_decay
#         lr = self.old_lr - lrd
#         for param_group in self.optimizer_D_A.param_groups:
#             param_group['lr'] = lr
#         for param_group in self.optimizer_D_B.param_groups:
#             param_group['lr'] = lr
#         for param_group in self.optimizer_G.param_groups:
#             param_group['lr'] = lr
#
#         print('update learning rate: %f -> %f' % (self.old_lr, lr))
#         self.old_lr = lr


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
from . import ssim

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

def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

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
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    return dice_total


class CycleGANModel(BaseModel):
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

        if opt.seg_norm == 'CrossEntropy':
            self.input_Seg_one = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,uncertainty=opt.uncertainty,leaky_relu = opt.leaky_relu,seg=False)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,uncertainty=opt.uncertainty,leaky_relu = opt.leaky_relu,seg=False)

        # self.netG_seg = networks.define_G(opt.input_nc_seg, opt.output_nc_seg,
        #                                 opt.ngf, opt.which_model_netSeg, opt.norm, not opt.no_dropout, self.gpu_ids,uncertainty=opt.uncertainty,leaky_relu = opt.leaky_relu,seg=True)

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
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                # self.load_network(self.netG_seg, 'Seg_A', which_epoch)
                if opt.seg_rec_loss or opt.seg_fakeMRI_realCT_loss:
                    self.load_network(self.netG_seg_mri, 'Seg_A_mri', which_epoch)

        # just_train_seg = True
        # if just_train_seg:
        #     for param in self.netG_A.parameters():
        #         param.requires_grad = False
        #     for param in self.netG_B.parameters():
        #         param.requires_grad = False
        #     for param in self.netD_A.parameters():
        #         param.requires_grad = False
        #     for param in self.netD_B.parameters():
        #         param.requires_grad = False

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.Wasserstein_Lossy:
                self.criterionGAN = networks.WassersteinLoss()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, target_real_label=1.0, target_fake_label=0.0)

            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            if opt.MIND_loss:
                self.criterionMIND = networks.MINDLoss(non_local_region_size=opt.non_local_region_size,
                                                       patch_size=opt.patch_size, neighbor_size=opt.neighbor_size,
                                                       gaussian_patch_sigma=opt.gaussian_patch_sigma,loss_type=opt.MIND_loss_type).cuda()

            # if self.opt.seg_rec_loss:
            #     self.criterionREC =  torch.nn.L1Loss()

            # initialize optimizers
            if not self.opt.separate_segmentation:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                if opt.seg_rec_loss or opt.seg_fakeMRI_realCT_loss:
                    self.optimizer_G = torch.optim.Adam(
                        itertools.chain(self.netG_A.parameters(),  self.netG_seg_mri.parameters(), self.netG_B.parameters()),
                        lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))

            if self.opt.ssim_fake_images_loss:
                self.ssim_loss = ssim.SSIM()
            # elif self.opt.separate_segmentation:
            #     self.optimizer_G = torch.optim.Adam(
            #         itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            #         lr=opt.lr, betas=(opt.beta1, 0.999))
            #     self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #     self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #     self.optimizer_seg = torch.optim.Adam(self.netG_seg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))



            # if self.opt.segmentation_discriminator:
            #     self.criterionSEG = nn.SmoothL1Loss(reduction='sum')
            # if self.opt.just_segmentation:
            #     for param in netG_A.parameters():
            #         param.requires_grad = False
            #     for param in netG_B.parameters():
            #         param.requires_grad = False
            #     for param in netD_A.parameters():
            #         param.requires_grad = False
            #     for param in netD_B.parameters():
            #         param.requires_grad = False

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        # networks.print_network(self.netG_seg)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')




        # early stopping
        # if self.opt.enable_early_stopping:
        #     print('in early stopping  &&&&')
        #     self.best_metric = {'G_A': float('inf'), 'G_B': float('inf'), 'D_A': float('inf'), 'D_B': float('inf'), 'seg': float('inf')}
        #     self.epochs_since_improvement = {'G_A': 0, 'G_B': 0, 'D_A': 0, 'D_B': 0, 'seg': 0}
        #     self.disable_training = {'G_A': 0, 'G_B': 0, 'D_A': 0, 'D_B': 0, 'seg': 0}
        #     self.stop_training = 0

    def get_enable_training_model(self):
        list_enable = []
        for key, value in self.disable_training.items():
            if value == 0:
                list_enable.append(key)
        return list_enable


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_Seg = input['Seg']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_Seg.resize_(input_Seg.size()).copy_(input_Seg)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.seg_norm == 'CrossEntropy':
            input_Seg_one = input['Seg_one']
            self.input_Seg_one.resize_(input_Seg_one.size()).copy_(input_Seg_one)


    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_Seg = Variable(self.input_Seg)
        if self.opt.seg_norm == 'CrossEntropy':
            self.real_Seg_one = Variable(self.input_Seg_one.long())

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

        if self.opt.identity > 0:
            # print('*********** in identity')
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.idt_B = self.netG_B.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def gradient_penalty(self,discriminator, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        # epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
        epsilon = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
        interpolates = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)
        d_interpolates = discriminator(interpolates).cuda()

        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(d_interpolates.size(),device='cuda'),
                                        create_graph=True, retain_graph=True)[0]
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty

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

        elif self.opt.Wasserstein_Lossy:
            real_logits = netD.forward(real)
            generated_logits = netD.forward(fake.detach())
            loss_D = self.criterionGAN(real_logits, generated_logits, discriminator=True)
            if self.opt.gradient_penalty:
                gradient_penalty_loss = self.gradient_penalty(netD, real, fake)
                loss_D = loss_D + self.opt.gradient_penalty_lambda * gradient_penalty_loss

        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) #* self.opt.lambda_D

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_D

    def backward_G(self,epoch=0):
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
            self.loss_G_A = self.criterionGAN(pred_fake, True) #* self.opt.lambda_D
        elif self.opt.Wasserstein_Lossy:
            self.loss_G_A = self.criterionGAN(generated_logits=pred_fake, discriminator=False) * self.opt.lambda_D
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)

        if not self.opt.Wasserstein_Lossy:
            self.loss_G_B = self.criterionGAN(pred_fake, True)* self.opt.lambda_D
        elif self.opt.Wasserstein_Lossy:
            self.loss_G_B = self.criterionGAN(generated_logits=pred_fake, discriminator=False) # * self.opt.lambda_D
        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        if self.opt.MIND_loss:
            self.loss_mind_A = 0
            self.loss_mind_B = 0
            if self.opt.MIND_sameModalityLoss:
                self.loss_mind_A += self.criterionMIND(self.fake_A,
                                                       self.real_A) * lambda_B * self.opt.lambda_mind * self.opt.MIND_sameModalityLossWeight
                self.loss_mind_B += self.criterionMIND(self.fake_B,
                                                       self.real_B) * lambda_A * self.opt.lambda_mind * self.opt.MIND_sameModalityLossWeight
            if self.opt.MIND_diffModalityLoss:
                self.loss_mind_A += self.criterionMIND(self.fake_B,
                                                       self.real_A) * lambda_B * self.opt.lambda_mind * self.opt.MIND_diffModalityLossWeight
                self.loss_mind_B += self.criterionMIND(self.fake_A,
                                                       self.real_B) * lambda_A * self.opt.lambda_mind * self.opt.MIND_diffModalityLossWeight
        else:
            self.loss_mind_A = 0
            self.loss_mind_B = 0

        if self.opt.lambda_cc > 0:
            self.loss_CC_A = 0
            self.loss_CC_B = 0
            if self.opt.CC_sameModalityLoss:
                self.loss_CC_A += networks.Cor_CoeLoss(self.fake_A,
                                                       self.real_A) * lambda_B * self.opt.lambda_cc * self.opt.MIND_sameModalityLossWeight
                self.loss_CC_B += networks.Cor_CoeLoss(self.fake_B,
                                                       self.real_B) * lambda_A * self.opt.lambda_cc * self.opt.MIND_sameModalityLossWeight
            if self.opt.CC_diffModalityLoss:
                self.loss_CC_A += networks.Cor_CoeLoss(self.fake_B,
                                                       self.real_A) * lambda_B * self.opt.lambda_cc * self.opt.MIND_diffModalityLossWeight
                self.loss_CC_B += networks.Cor_CoeLoss(self.fake_A,
                                                       self.real_B) * lambda_A * self.opt.lambda_cc * self.opt.MIND_diffModalityLossWeight

        else:
            self.loss_CC_A = 0
            self.loss_CC_B = 0





        # Segmentation loss
        # self.seg_fake_B = self.netG_seg.forward(self.fake_B)
        # if self.opt.seg_norm == 'DiceNorm':
        #     self.loss_seg = dice_loss_norm(self.seg_fake_B, self.real_Seg)
        #     self.loss_seg = self.loss_seg
        # elif self.opt.seg_norm == 'CrossEntropy':
        #     arr = np.array(self.opt.crossentropy_weight)
        #     weight = torch.from_numpy(arr).cuda().float()
        #
        #     self.loss_seg = CrossEntropy2d(self.seg_fake_B, self.real_Seg_one, weight=weight)
        #
        #     if self.opt.segmentation_discriminator:
        #         self.seg_real_B = self.netG_seg.forward(self.real_B)
        #         huber_loss = self.criterionSEG(self.seg_real_B, self.seg_fake_B)
        #         # print('huber_loss: ', huber_loss)
        #         self.loss_seg = self.loss_seg + huber_loss








        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A +
                       self.loss_idt_B + self.loss_mind_A + self.loss_mind_B + self.loss_CC_A +
                           self.loss_CC_B)

        if self.opt.ssim_fake_images_loss:
            self.loss_ssim_B = (1 - self.ssim_loss(self.fake_B, self.real_B)) * self.opt.weight_ssim_loss
            self.loss_ssim_A = (1 - self.ssim_loss(self.fake_A, self.real_A)) * self.opt.weight_ssim_loss
            # print('ssim_A, ssim_B', ssim_A, ssim_B)
            self.loss_G += (self.loss_ssim_B + self.loss_ssim_A)

        if self.opt.direct_loss:
            # Forward direct loss
            self.cycled_A = self.netG_A.forward(self.fake_B)
            self.loss_cycled_A = self.criterionCycle(self.cycled_A, self.real_A)  # * lambda_A
            # print('self.loss_cycled_A : ',self.loss_cycled_A )
            self.cycled_B = self.netG_B.forward(self.fake_A)
            self.loss_cycled_B = self.criterionCycle(self.cycled_B, self.real_B)  # * lambda_B
            self.loss_G = self.loss_G + self.loss_cycled_B + self.loss_cycled_A


        if self.opt.seg_fakeMRI_realCT_loss:
            self.seg_rec_A = self.netG_seg_mri.forward(self.rec_A)
            self.seg_real_A = self.netG_seg_mri.forward(self.real_A)
            self.seg_fake_A = self.netG_seg_mri.forward(self.fake_A)


            arr = np.array(self.opt.crossentropy_weight)
            weight = torch.from_numpy(arr).cuda().float()
            self.loss_seg_real_mri = CrossEntropy2d(self.seg_real_A, self.real_Seg_one, weight=weight)
            self.loss_seg_rec_mri = CrossEntropy2d(self.seg_rec_A, self.real_Seg_one, weight=weight)



            self.loss_G = self.opt.weight_segmentation_in_GAN*( self.loss_seg_rec_mri + self.loss_seg_real_mri) + self.loss_G


        self.loss_G.backward()






    def optimize_parameters(self,data_number=0,epoch=0):
        if not self.opt.enable_early_stopping:
            # forward
            self.forward()
            # G_A and G_B
            self.optimizer_G.zero_grad()
            # if self.opt.separate_segmentation :
            #     self.optimizer_seg.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # if self.opt.separate_segmentation :
            #     self.optimizer_seg.step()
            # D_A
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

        error_list = [('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                      ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),
                      ]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.item()
            idt_B = self.loss_idt_B.item()
            error_list.append(('idt_B', idt_B))
            error_list.append(('idt_A', idt_A))
            # return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
            #                     ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B), ('Seg', Seg_B)])
        # else:
        #     return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
        #                         ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),
        #                         ('Seg', Seg_B)])
        if self.opt.seg_rec_loss:
            loss_seg_rec_mri = self.loss_seg_mri.item()
            error_list.append(('seg_rec_mri', loss_seg_rec_mri))
        if self.opt.seg_fakeMRI_realCT_loss:

            error_list.append(('loss_seg_rec_mri', self.loss_seg_rec_mri.item()))
            error_list.append(('loss_seg_real_mri', self.loss_seg_real_mri.item()))

        if self.opt.ssim_fake_images_loss:
            error_list.append(('ssim_A', self.loss_ssim_A.item()))
            error_list.append(('ssim_B', self.loss_ssim_B.item()))
        if self.opt.MIND_loss:
            error_list.append(('mind_A', self.loss_mind_A.item()))
            error_list.append(('mind_B', self.loss_mind_B.item()))
        if self.opt.lambda_cc>0:
            error_list.append(('cc_A', self.loss_CC_A.item()))
            error_list.append(('cc_B', self.loss_CC_B.item()))
        return error_list

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        # seg_B = util.tensor2seg(torch.max(self.seg_fake_B.data,dim=1,keepdim=True)[1])
        # manual_B = util.tensor2seg(torch.max(self.real_Seg.data,dim=1,keepdim=True)[1])
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        # self.save_network(self.netG_seg, 'Seg_A', label, self.gpu_ids)

    def save_individual_model(self, model_name, label):
        if model_name == 'G_A':
            self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        elif model_name == 'D_A':
            self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        elif model_name == 'G_B':
            self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        elif model_name == 'D_B':
            self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        # elif model_name == 'Seg_A':
        #     self.save_network(self.netG_seg, 'Seg_A', label, self.gpu_ids)

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

        # if not self.disable_training['seg']:
        #     self.check_eraly_stopping('Seg_A', self.netG_seg, self.loss_seg, self.opt.min_delta_seg, epoch)

    def check_eraly_stopping(self,name,model,loss,delta,epoch):
        print('in check early stopping ', name)
        # if self.best_metric[name] - loss >=  delta:
        #     self.best_metric[name] = loss
        #     self.epochs_since_improvement[name] = 0
        #     # Save the best model
        #     self.save_individual_model(name,f'best_{name}')
        # else:
        #     print('I am in plus for ', name)
        #     self.epochs_since_improvement[name] += 1
        #
        # patience = 0
        # if 'D' in name:
        #     if self.epochs_since_improvement[name] >= self.opt.patience_D and epoch<60:
        #         self.opt.patience_D = self.opt.patience_D + 5
        #
        #     patience = self.opt.patience_D
        # elif 'G_':
        #     if self.epochs_since_improvement[name] >= self.opt.patience_G and epoch<60:
        #         self.opt.patience_G = self.opt.patience_G + 5
        #
        #     patience = self.opt.patience_G
        # elif 'seg':
        #     if self.epochs_since_improvement[name] >= self.opt.patience_seg and epoch<60:
        #         self.opt.patience_seg = self.opt.patience_seg + 5
        #
        #     patience = self.opt.patience_seg



        # if self.epochs_since_improvement[name] >= patience:
        #     print(f"Early stopping for {name}in epoch {epoch} patience {patience}!")
        #     self.save_individual_model(name, f'best_{name}')
        #     for param in model.parameters():
        #         param.requires_grad = False
        #     self.disable_training[name] = 1

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
