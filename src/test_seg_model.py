import numpy as np
import torch
import os
from monai.data import  decollate_batch
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
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    Compose,
)


class TestSEGModel(BaseModel):
    # def name(self):
    #     return 'TestSEGModel'

    def initialize(self, opt):
        self.option = opt
        BaseModel.initialize(self, opt)

        self.netG_seg = networks.define_G(self.option, seg_net=True)
        self.load_network(self.netG_seg, 'Seg_B', self.option.which_epoch)

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        self.real_B = input['CT'].to(self.opt.device)
        print('realB in set_input ', self.real_B.shape)
        self.real_Seg = input['label'].to(self.opt.device)
        print('real seg in set_input before onehot ', np.shape(self.real_Seg), flush=True)
        self.real_Seg = one_hot(self.real_Seg, num_classes=self.opt.num_classes, dim=1)
        print('real seg in set_input ', self.real_Seg.shape)


    def inference(self):
        def _compute():
            return sliding_window_inference(
                inputs=self.real_B,
                roi_size=self.option.roi_size,
                sw_batch_size=1,
                predictor= self.netG_seg ,
                overlap=0.5,
            )

        if self.option.VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute()
        else:
            return _compute()

    def calculate_inference(self):

        if self.option.calculate_uncertainty:
            self.get_monte_carlo_predictions()
        else:
            self.pred_seg = self.inference()

    def get_current_visuals(self):

        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=True)])
        self.pred = [post_trans(i) for i in decollate_batch(self.pred_seg)]
        self.real = [post_trans(i) for i in decollate_batch(self.real_Seg)]

        return {'seg_B': self.pred, 'real_seg': self.real, 'variance': self.variance, 'entropy': self.entropy}

    def eval(self):
        self.netG_seg.eval()
        if self.option.calculate_uncertainty:
            print('in uncertainty')
            self.enable_dropout(self.netG_seg)


    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print('******', m.__class__.__name__)
                m.train()

    def get_monte_carlo_predictions(self):
        """ Function to get the monte-carlo samples and uncertainty estimates
        through multiple forward passes

        """
        n_samples = self.option.uncertainty_num_samples
        sample_predictions = []

        for j in range(n_samples):
            output = self.inference()
            sample_predictions.append(output)

        prediction = torch.stack(sample_predictions)
        print('prediction', np.shape(prediction))

        # Calculating mean across multiple MCD forward passes
        mean_predictions = torch.mean(prediction, dim=0)  # shape (n_samples, n_classes)
        self.pred_seg = mean_predictions
        print('mean prediction ', np.shape(mean_predictions))

        # Calculating variance across multiple MCD forward passes
        self.variance = torch.var(prediction, axis=0)  # shape (n_samples, n_classes)
        print('variance', np.shape(self.variance))

        # Calculate the entropy of a probability distribution
        self.entropy = -(mean_predictions * torch.log(mean_predictions + 1e-9)).sum(dim=1)
        print('entropy', np.shape(self.entropy))












