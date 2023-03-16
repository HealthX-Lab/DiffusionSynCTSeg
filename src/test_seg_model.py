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
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    Compose,
)


class TestSEGModel(BaseModel):
    def name(self):
        return 'TestSEGModel'

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
            self.pred_seg = sliding_window_inference(
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

    def get_current_visuals(self):
        print('self.real_Seg.data', np.shape(self.real_Seg.data))
        print('self.pred_seg', np.shape(self.pred_seg.data))
        print('self.real_Seg.data2', np.shape(self.real_Seg))
        print('self.pred_seg2', np.shape(self.pred_seg))
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=True)])
        self.pred_seg = post_trans(self.pred_seg)
        print('self.pred_seg3', np.shape(self.pred_seg))
        return {'seg_B':self.pred_seg, 'real_seg':self.real_Seg}
        # return self.pred_seg, self.real_Seg

    def eval(self):
        self.netG_seg












