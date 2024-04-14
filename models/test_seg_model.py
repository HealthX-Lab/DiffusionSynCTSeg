from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch

class TestSegModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      self.gpu_ids)

        self.netG_seg = networks.define_G(opt.input_nc_seg, opt.output_nc_seg,
                                        opt.ngf, opt.which_model_netSeg, opt.norm, not opt.no_dropout, self.gpu_ids)



        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G_A', which_epoch)
        self.load_network(self.netG_seg, 'Seg_A', which_epoch)

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

    def test(self):
        entropy_sum = 0.0
        if self.opt.uncertainty:
            self.real_A = Variable(self.input_A)
            entropy_map = torch.zeros(self.real_A.shape[-2:])
            num_samples=20
            for i in range(0,num_samples):

                self.fake_B = self.netG_seg.forward(self.real_A)
                probs = torch.softmax(self.fake_B, dim=1)  # apply softmax function
                entropy = -(probs * torch.log2(probs + 1e-12)).sum(dim=1)  # calculate entropy
                entropy_map += entropy.mean(dim=0).cpu()
            entropy_map /= num_samples
            self.entropy_map = entropy_map
        else:
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG_seg.forward(self.real_A)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2seg(torch.max(self.fake_B.data,dim=1,keepdim=True)[1])
        if self.opt.uncertainty:
            self.entropy_map = self.entropy_map.detach()
            self.entropy_map = self.entropy_map.unsqueeze(0).unsqueeze(0)
            entropy = util.tensor2im(self.entropy_map)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('entropy', entropy)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
