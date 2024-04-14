from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_Seg = self.Tensor(nb, opt.output_nc_seg, size, size)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      self.gpu_ids)

        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,uncertainty=opt.uncertainty)
        #
        self.netG_seg = networks.define_G(opt.input_nc_seg, opt.output_nc_seg,
                                          opt.ngf, opt.which_model_netSeg, opt.norm, not opt.no_dropout, self.gpu_ids)
        which_epoch = opt.which_epoch
        self.load_network(self.netG_A, 'G_A', which_epoch)
        self.load_network(self.netG_B, 'G_B', which_epoch)
        self.load_network(self.netG_seg, 'Seg_A', which_epoch)

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG_A)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

        input_B = input['B']
        self.input_B.resize_(input_B.size()).copy_(input_B)

        input_Seg = input['Seg']
        self.input_Seg.resize_(input_Seg.size()).copy_(input_Seg)



    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)


        self.input_Seg = Variable(self.input_Seg)
        self.fake_seg = self.netG_seg.forward(self.fake_B)

        self.real_seg = self.netG_seg.forward(self.real_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)

        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)

        input_Seg = util.tensor2im(self.input_Seg.data)

        fake_seg = util.tensor2seg(torch.max(self.fake_seg.data, dim=1, keepdim=True)[1])
        real_seg = util.tensor2seg(torch.max(self.real_seg.data, dim=1, keepdim=True)[1])

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                            ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                            ('fake_seg', fake_seg),('real_seg', real_seg), ('input_seg', input_Seg)])

        # return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
