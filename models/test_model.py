from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torch.nn.functional as F


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
        print('opt.which_model_netG,opt.which_model_netSeg',opt.which_model_netG,opt.which_model_netSeg)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, True,
                                      self.gpu_ids)#not opt.no_dropout

        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, True, self.gpu_ids,uncertainty=opt.uncertainty)#not opt.no_dropout
        #
        self.netG_seg = networks.define_G(opt.input_nc_seg, opt.output_nc_seg,
                                          opt.ngf, opt.which_model_netSeg, opt.norm, True, self.gpu_ids)#not opt.no_dropout



        if opt.which_epoch <= opt.max_epoch:
            param_A = opt.which_epoch
            param_B = opt.which_epoch
            param_seg = opt.which_epoch
        if opt.which_epoch == opt.max_epoch + 1:
            param_A = 'best_MI_fake_B'
            param_B = 'best_MI_fake_A'
            param_seg = 'best_dice_seg_real'
        elif opt.which_epoch == opt.max_epoch + 2:
            param_A = 'best_mse_fake_B'
            param_B = 'best_mse_fake_A'
            param_seg = 'best_dice_seg_real'
        elif opt.which_epoch == opt.max_epoch + 3:
            param_A = 'best_ssim_fake_B'
            param_B = 'best_ssim_fake_A'
            param_seg = 'best_dice_seg_real'

        self.load_network(self.netG_A, 'G_A', param_A)#which_epoch)
        self.load_network(self.netG_B, 'G_B', param_B)#which_epoch)
        self.load_network(self.netG_seg, 'Seg_A', param_seg)#which_epoch)
        self.netG_A.eval()
        self.netG_B.eval()
        self.netG_seg.eval()

        print('-----------------------------------------------')
        self.image_array = np.zeros((30, 136, 156), dtype=np.float32)

        self.mse_A = 0
        self.mse_B = 0
        self.mse_fake_A = 0
        self.mse_fake_B = 0
        self.seg_rec = []
        self.seg_fake = []
        self.seg_real = []
        self.seg_real_IOU = 0
        self.var_fake_A = 0
        self.var_fake_B = 0
        self.var_rec_A = 0
        self.var_rec_B = 0
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
        self.mse_A = 0
        self.mse_B = 0
        self.mse_fake_A = 0
        self.mse_fake_B = 0
        self.seg_real_IOU = 0
        self.var_fake_A = 0
        self.var_fake_B = 0
        self.var_rec_A = 0
        self.var_rec_B = 0
        self.var_seg_real = 0
        self.var_seg_fake = 0
        self.number_of_images = self.number_of_images + 1
        self.precision = 0
        self.recall = 0
        self.specificity = 0

        self.ssim_value_A = 0
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


    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths_A = input['A_paths']
        self.image_paths_B = input['B_paths']
        self.image_paths_seg = input['Seg_paths']

        input_B = input['B']
        self.input_B.resize_(input_B.size()).copy_(input_B)

        input_Seg = input['Seg']
        self.input_Seg.resize_(input_Seg.size()).copy_(input_Seg)

        input_Seg_one = input['Seg_one']
        self.input_Seg_one = self.Tensor(1,2,256,256)
        self.input_Seg_one.resize_(input_Seg_one.size()).copy_(input_Seg_one)
        # print('one: ',self.input_Seg_one.size() )





    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        models = [self.netG_A, self.netG_B, self.netG_seg]
        for i in range(0,len(models)):
            for m in models[i].modules():
                # print('*&*&*&', m.__class__.__name__)
                if m.__class__.__name__.startswith('Dropout'):
                    # print('******', m.__class__.__name__)
                    m.train()

    def compute_entropy(self, output):
        p = torch.softmax(output, dim=1)
        return -(p * torch.log(p)).sum(dim=1)

    def compute_output(self, outputs):
        mean = {key: None for key in outputs.keys()}
        var = {key: None for key in outputs.keys()}
        heatmap = {key: None for key in outputs.keys()}
        uncertainty_map = {key: None for key in outputs.keys()}
        confidence_map = {key: None for key in outputs.keys()}
        entropy_map = {key: None for key in outputs.keys()}


        for key in outputs.keys():
            # print(key)
            if 'seg' in key:
                output_probability = torch.stack(outputs[key])
                mean[key] = output_probability.mean(dim=0)
                var[key] = output_probability.var(dim=0)




            else:
                stacked_output = torch.stack(outputs[key])
                mean[key] = stacked_output.mean(dim=0)
                var[key] = stacked_output.var(dim=0)
                # print('var ', var[key].size())
            '''in summary, a heatmap in the context of MC Dropout and segmentation would show the most probable class for 
            each pixel, as determined by averaging over multiple runs with dropout enabled.'''
            if 'seg' in key:
                heatmap[key] = torch.argmax(mean[key], dim=1)
                # print('heatmap shape ', np.shape(heatmap[key]))
                '''Lower values typically indicate high certainty, and higher values indicate more uncertainty. These are
                 particularly useful in scenarios where the model's prediction needs to be trusted and acted upon, like 
                 medical image analysis.'''
                uncertainty_map[key] =  var[key].max(dim=1)[0].sqrt()


                '''A confidence map, on the other hand, visualizes the model's confidence in its predictions. Each pixel 
                in the confidence map corresponds to a pixel in the image, and its intensity (or color) indicates how 
                confident the model is that its prediction for that pixel is correct.'''
                confidence_map[key] = mean[key].max(dim=1)[0]
                '''a measure of the model's uncertainty about its predictions. Higher entropy indicates higher uncertainty, 
                while lower entropy indicates lower uncertainty.'''
                entropy_map[key] = self.compute_entropy(mean[key])
                if key == 'seg_real':
                    if self.number_of_images < 35:
                        self.cal_manual_dice(outputs[key])
                        self.cal_manual_iou(outputs[key])
                    else:
                        self.tn1, self.fp1, self.fn1, self.tp1 = [65536,0,0,0]
                        self.tn2, self.fp2, self.fn2, self.tp2 = [65536,0,0,0]
                        self.mean_pure_dice = self.mean_dice_gamma_1 = self.mean_dice_gamma_2 = torch.tensor(1.0,
                                                                                                             dtype=torch.float32).item()
                        self.one_pure_dice = self.one_dice_gamma_1 = self.one_dice_gamma_2 = torch.tensor(1.0,
                                                                                                          dtype=torch.float32).item()
                        self.mean_pure_iou = self.mean_iou_gamma_1 = self.mean_iou_gamma_2 = torch.tensor(1.0,
                                                                                                          dtype=torch.float32)
                        self.one_pure_iou = self.one_iou_gamma_1 = self.one_iou_gamma_2 = torch.tensor(1.0,
                                                                                                       dtype=torch.float32)

                # print('*******************************************************')
            else:
                uncertainty_map[key] = var[key].sqrt()[0]


        return mean, var, heatmap, uncertainty_map, confidence_map, entropy_map

    def test(self):
        outputs = {'fake_A':[],'fake_B':[],'rec_A':[],'rec_B':[],'fake_seg':[],'seg_real':[],'seg_rec':[]}
        if self.opt.MC_uncertainty:

            for i in range (0,self.opt.num_samples_uncertainty):
                # print(i , 'in test uncertanty')
                self.real_A = Variable(self.input_A)
                outputs['fake_B'].append(self.netG_A.forward(self.real_A))
                outputs['rec_A'].append(self.netG_B.forward(outputs['fake_B'][i]))

                self.real_B = Variable(self.input_B)
                outputs['fake_A'].append(self.netG_B.forward(self.real_B))
                outputs['rec_B'].append(self.netG_A.forward(outputs['fake_A'][i]))

                self.input_Seg = Variable(self.input_Seg)
                outputs['fake_seg'].append(F.softmax(self.netG_seg.forward(outputs['fake_B'][i]), dim=1))

                outputs['seg_real'].append(F.softmax(self.netG_seg.forward(self.real_B), dim=1))

                outputs['seg_rec'].append(F.softmax(self.netG_seg.forward(outputs['rec_B'][i]), dim=1))

                for key in outputs.keys():
                    outputs[key][-1] = outputs[key][-1].detach().cpu()

            self.mean, self.var, self.heatmap, self.uncertainty_map, self.confidence_map, self.entropy_map= self.compute_output(outputs)
            self.model_images = outputs
            if self.number_of_images < 30:
                self.set_image(self.number_of_images, self.crop_images(self.mean['seg_real']))
            self.set_coef()
            if self.number_of_images % 41 == 40:
                self.set_3Dcoef()

        else:
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG_A.forward(self.real_A)
            self.rec_A = self.netG_B.forward(self.fake_B)

            self.real_B = Variable(self.input_B)
            self.fake_A = self.netG_B.forward(self.real_B)
            self.rec_B = self.netG_A.forward(self.fake_A)


            self.input_Seg = Variable(self.input_Seg)
            self.fake_seg = self.netG_seg.forward(self.fake_B)

            self.real_seg = self.netG_seg.forward(self.real_B)

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
        self.processed_dice3d(y_pred1, y_pred2)

        return pure_dice, dice_gamma_1, dice_gamma_2
    def processed_dice3d(self, y_pred1, y_pred2):
        y_true = self.input_Seg[:, 1, :, :].view(-1).detach().cpu().numpy()
        y_pred1 = y_pred1.view(-1).detach().cpu().numpy()
        y_pred2 = y_pred2.view(-1).detach().cpu().numpy()

        self.tn1, self.fp1, self.fn1, self.tp1 = confusion_matrix(y_true, y_pred1, labels=[0, 1]).ravel()
        self.tn2, self.fp2, self.fn2, self.tp2 = confusion_matrix(y_true, y_pred2, labels=[0, 1]).ravel()




    def calculate_ssim(self, image1, image2):
        import torch
        import torchmetrics
        import torchvision.transforms.functional as F

        sigma = 2
        kernel_size = 7

        # Applying Gaussian blur
        image1 = F.gaussian_blur(image1, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        image2 = F.gaussian_blur(image2, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        device = image1.device
        # Initialize SSIM metric
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(kernel_size=11, data_range=1.0,
                                                                    gaussian_kernel=True, sigma=1).to(device)

        # Calculate SSIM
        ssim_value = ssim_metric(image1, image2)
        return ssim_value.item()
    def crop_images(self, seg):
        # print('****&*&*',np.shape(seg))
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

    # get image paths
    def get_image_paths(self):
        return self.image_paths_A, self.image_paths_B, self.image_paths_seg

    def get_current_visuals(self):
        if self.opt.MC_uncertainty:
            visuals = {}
            for key in self.model_images.keys():
                visuals_images = []
                for i in range(self.opt.num_samples_uncertainty):
                    # print('*******',key)
                    if 'seg' not in key:
                        visuals_images.append(util.tensor2im(self.model_images[key][i].data))
                    else:
                        visuals_images.append(util.tensor2seg(torch.max(self.model_images[key][i].data, dim=1, keepdim=True)[1]))


                visuals[key] = visuals_images


            heatmap = {key: util.tensor2seg(self.heatmap[key]) for key in self.heatmap.keys() if 'seg' in key}

            uncertainty_map = {key: util.tensor2map(self.uncertainty_map[key]) for key in self.uncertainty_map.keys()}

            confidence_map = {key: util.tensor2map(self.confidence_map[key]) for key in self.confidence_map.keys() if 'seg' in key}

            entropy_map = {key: util.tensor2map(self.entropy_map[key]) for key in self.entropy_map.keys() if 'seg' in key}

            var_map = {key: util.tensor2map(self.var[key]) for key in self.var.keys()  if 'seg' not in key}



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



        else:
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            rec_A = util.tensor2im(self.rec_A.data)

            real_B = util.tensor2im(self.real_B.data)
            fake_A = util.tensor2im(self.fake_A.data)
            rec_B = util.tensor2im(self.rec_B.data)

            input_Seg = util.tensor2im(self.input_Seg.data)


            fake_seg = util.tensor2seg(self.fake_seg.data[:,0,:,:])

            print('seg real shape',self.real_seg.shape)

            real_seg = util.tensor2seg(self.fake_seg.data[:, 1, :, :])

            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                                ('fake_seg', fake_seg),('real_seg', real_seg), ('input_seg', input_Seg)])

        # return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def avg_uncertainty(self, uncertainty_per_pixel):
        total_uncertainty = np.mean(uncertainty_per_pixel.detach().cpu().numpy())
        return total_uncertainty

    def set_coef(self):

        self.var_fake_A_2D = self.avg_uncertainty(self.uncertainty_map['fake_A'])
        self.var_fake_B_2D = self.avg_uncertainty(self.uncertainty_map['fake_B'])
        self.var_rec_A_2D = self.avg_uncertainty(self.uncertainty_map['rec_A'])
        self.var_rec_B_2D = self.avg_uncertainty(self.uncertainty_map['rec_B'])
        self.var_seg_real_2D = self.avg_uncertainty(self.uncertainty_map['seg_real'])
        self.var_seg_fake_2D = self.avg_uncertainty(self.uncertainty_map['fake_seg'])
        self.ssim_value_A_2D = ssim(self.mean['fake_A'].cuda(), self.real_A, data_range=1.0, size_average=True).item()
        self.ssim_value_B_2D = ssim(self.mean['fake_B'].cuda(), self.real_B, data_range=1.0, size_average=True).item()
        self.new_ssim_B = self.calculate_ssim(self.mean['fake_B'].cuda(), self.real_B.cuda())
        self.new_ssim_A = self.calculate_ssim(self.mean['fake_A'].cuda(), self.real_A.cuda())

        self.mse_A_2D = self.cal_mse(self.mean['rec_A'], self.real_A)
        self.mse_B_2D = self.cal_mse(self.mean['rec_B'], self.real_B)
        self.mse_fake_A_2D = self.cal_mse(self.mean['fake_A'], self.real_A)
        self.mse_fake_B_2D = self.cal_mse(self.mean['fake_B'], self.real_B)
        self.seg_real_IOU_2D = self.get_IOU(self.heatmap['seg_real'], self.input_Seg)

        self.seg_rec_2D = self.dice_coef(self.heatmap['seg_rec'], self.input_Seg, smooth=1)
        self.seg_fake_2D = self.dice_coef(self.heatmap['fake_seg'], self.input_Seg, smooth=1)
        self.seg_real_2D = self.dice_coef(self.heatmap['seg_real'], self.input_Seg, smooth=1)
        self.get_confusion_matrix(self.heatmap['seg_real'], self.input_Seg, 'real')
        self.recal_2D = self.cal_recal()
        self.specificity_2D = self.cal_specificity()
        self.percision_2D = self.cal_percision()

        self.hist_similarity_B = self.compare_histograms(self.mean['fake_B'], self.real_B)
        self.hist_similarity_A = self.compare_histograms(self.mean['fake_A'], self.real_A)






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

                # 'mse_A': self.mse_A,
                'mse_rec_A_2D': self.mse_A_2D,
                # 'mse_B': self.mse_B,
                'mse_rec_B_2D': self.mse_B_2D,
                # 'mse_fake_A': self.mse_fake_A,
                'mse_fake_A_2D': self.mse_fake_A_2D,
                # 'mse_fake_B': self.mse_fake_B,
                'mse_fake_B_2D': self.mse_fake_B_2D,
                # 'seg_real_IOU': self.seg_real_IOU,
                'seg_real_IOU_2D': self.seg_real_IOU_2D.item(),
                'dice_seg_rec_2D': self.seg_rec_2D,
                'dice_seg_fake_2D': self.seg_fake_2D,
                'dice_seg_real_2D': self.seg_real_2D,


                'ssim_value_fake_B_2D': self.ssim_value_B_2D,
                'new_ssim_A': self.new_ssim_A,
                'new_ssim_B': self.new_ssim_B,
                # 'ssim_value_A': self.ssim_value_A,
                'ssim_value_fake_A_2D': self.ssim_value_A_2D,
                # 'var_seg_fake': self.var_seg_fake,
                'var_seg_fake_2D': self.var_seg_fake_2D,
                # 'var_seg_real': self.var_seg_real,
                'var_seg_real_2D': self.var_seg_real_2D,
                # 'var_rec_B': self.var_rec_B,
                'var_rec_B_2D': self.var_rec_B_2D,
                # 'var_rec_A': self.var_rec_A,
                'var_rec_A_2D': self.var_rec_A_2D,
                # 'var_fake_B': self.var_fake_B,
                'var_fake_B_2D': self.var_fake_B_2D,
                # 'var_fake_A': self.var_fake_A,
                'var_fake_A_2D': self.var_fake_A_2D,
                'percision_2D': self.percision_2D,
                'specificity_2D': self.specificity_2D,
                'recal_2D': self.recal_2D,

                'corr_hist_A': self.hist_similarity_A,
                'corr_hist_B': self.hist_similarity_B,

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
        ground_truth = ground_truth[:,1,:,:]
        ground_truth = ground_truth.detach().cpu()
        intersection = torch.sum(predicted * ground_truth)
        dice = (2.0 * intersection + smooth) / (torch.sum(predicted) + torch.sum(ground_truth) + smooth)
        return dice.item()

    def get_IOU(self, predicted, ground_truth):
        ground_truth = ground_truth[:,1,:,:]
        ground_truth = ground_truth.detach().cpu()
        intersection = torch.sum(predicted * ground_truth)
        union = (predicted.bool() | ground_truth.bool()).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    def cal_mse(self, input, target):
        target = target.detach().cpu()
        mse = torch.mean((input - target) ** 2)
        return mse.item()

    def get_confusion_matrix(self, y_pred, y_true,key):
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

    def set_3Dcoef(self):

        if 2*self.TP + self.FP + self.FN:
            self.dice = 2 * self.TP / ( 2*self.TP + self.FP + self.FN)
        if self.TP + self.FP + self.FN:
            self.iou = self.TP / (self.TP + self.FP + self.FN)
        if self.TP + self.FN:
            self.recal = self.TP / (self.TP + self.FN)
        if self.TN + self.FP:
            self.specificity =self.TN / (self.TN + self.FP)
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

    def set_image(self, i, seg):
        self.image_array[i] = seg
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
    def get_image(self):
        return self.image_array
    def get_name(self):
        return  self.image_paths_A[0].split('/')[-1]





