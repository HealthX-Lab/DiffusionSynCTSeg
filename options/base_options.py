import argparse
import os
from util import util
import torch
from distutils.util import strtobool


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', default='./datasets/yh',
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--input_nc_seg', type=int, default=1,
                                 help='# of input image channels for segmentation')
        self.parser.add_argument('--output_nc_seg', type=int, default=2,
                                 help='# of output image channels for segmentation')

        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')

        self.parser.add_argument('--mode', type=str, default='2d',
                                 help='2d or 3d segmentation model ')
        self.parser.add_argument('--Depth', type=int, default=32, help='# depth for 3D segmentation')
        self.parser.add_argument('--serial_batches', type=bool, default=False,
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')

        self.parser.add_argument('--which_model_netD', type=str, default='basic',
                                 help='selects model to use for netD n_layers or basic')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks',
                                 help='selects model to use for netG swin  R2AttU_Net unet_128  resnet_9blocks')
        self.parser.add_argument('--which_model_netSeg', type=str, default='R2AttU_Net',
                                 help=' swin unet_128,  resnet_9blocks, R2AttU_Net, AttU_Net, R2U_Net selects model to do segmentation netSeg')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--name', type=str, default='zero_centered_MIND_CC_cyclegan_segmentation',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--test_seg_output_dir', type=str,
                                 default='./MIND_CC_cyclegan_boundary_loss_segmentation',
                                 help='save test sege output results')

        self.parser.add_argument('--dataset_mode', type=str, default='yh_seg',
                                 help='chooses how datasets are loadfed. [unaligned | aligned | single | yh | yh_seg| yh_test_seg | yh_seg]')
        self.parser.add_argument('--model', type=str, default='cycle_seg',
                                 help='chooses which model to use. sb (for UNSB) | cut (for CUT model) |  cycle_seg (for cyclegan with segmentaiton) | test (CycleGAN test)  | TestGANModel | cycle_gan')
        self.parser.add_argument('--yh_run_model', type=str, default='Train',
                                 help='chooses which Test, Train,  TestSeg')
        self.parser.add_argument('--identity', type=float, default=0.5,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--lambda_D', type=float, default=1.0, help='weight for discriminator loss')

        self.parser.add_argument('--train_dataset_paired_80_120', type=bool, default=False,
                                 help='train_dataset_paired_80_120')
        self.parser.add_argument('--train_dataset_IXI_neuro_80_120', type=bool, default=True,
                                 help='train_dataset_IXI_neuro_80_120')
        self.parser.add_argument('--train_dataset_neuro_80_120', type=bool, default=False,
                                 help='train_dataset_neuro_80_120')
        self.parser.add_argument('--train_dataset_neuro_all_slices', type=bool, default=False,
                                 help='train_dataset_neuro_all_slices')
        self.parser.add_argument('--train_dataset_finetune', type=bool, default=False, help='train_dataset_finetune')
        self.parser.add_argument('--train_dataset_finetune_06', type=bool, default=False,
                                 help='train_dataset_finetune_06')
        self.parser.add_argument('--train_dataset_neuro_85_100', type=bool, default=False,
                                 help='train_dataset_neuro_85_100')
        self.parser.add_argument('--train_dataset_IXI_neuro_all_slice', type=bool, default=False,
                                 help='train_dataset_IXI_neuro_all_slice')
        self.parser.add_argument('--normalize_dataset_80_120', type=bool, default=False, help='data between 0-1')

        self.parser.add_argument('--test_dataset_iDB_10_500_80_120', default='False',
                                 help='test_dataset_iDB_10_500_80_120', type=lambda x: bool(strtobool(x)))
        self.parser.add_argument('--test_dataset_train_10_500_80_120', type=lambda x: bool(strtobool(x)),
                                 default="False", help='test_dataset_train_10_500_80_120')
        self.parser.add_argument('--test_dataset_iDB_10_500_85_100', type=lambda x: bool(strtobool(x)), default="False",
                                 help='test_dataset_iDB_10_500_85_100')
        self.parser.add_argument('--test_dataset_iDB_10_400_80_120', type=lambda x: bool(strtobool(x)), default='False',
                                 help='test_dataset_iDB_10_400_80_120')
        self.parser.add_argument('--test_dataset_iDB_10_300_80_120', type=lambda x: bool(strtobool(x)), default='False',
                                 help='test_dataset_iDB_10_300_80_120')
        self.parser.add_argument('--test_dataset_iDB_10_200_80_120', type=lambda x: bool(strtobool(x)), default='False',
                                 help='test_dataset_iDB_10_200_80_120')
        self.parser.add_argument('--test_dataset_iDB_10_100_80_120', type=lambda x: bool(strtobool(x)), default='False',
                                 help='test_dataset_iDB_10_100_80_120')
        self.parser.add_argument('--test_dataset_iDB_10_50_80_120', type=lambda x: bool(strtobool(x)), default='False',
                                 help='test_dataset_iDB_10_50_80_120')
        self.parser.add_argument('--test_iDB_normalize_dataset_10_500_80_120', type=lambda x: bool(strtobool(x)),
                                 default=False, help='data between 0-1')

        self.parser.add_argument('--path_images', type=str,
                                 default='./model_outputs/Output_save/final2D', help='')
        self.parser.add_argument('--path_csv', type=str, default='./model_outputs/csvfiles/final2D',
                                 help='')
        self.parser.add_argument('--folder_name', type=str, default='new_data_preprocess', help='')
        self.parser.add_argument('--len_dataset', type=int, default=0, help=' len_dataset ')
        self.parser.add_argument('--max_epoch', type=int, default=100, help=' max_epoch ')
        self.parser.add_argument('--print_images_with_uncertainty', type=lambda x: bool(strtobool(x)), default='False',
                                 help='print_images_with_uncertainty')
        self.parser.add_argument('--attribute', type=str, default='simple', help=' attribute ')

        self.parser.add_argument('--add_quantum_noise', type=bool, default=False,
                                 help='add quantum noise to fake ct images before segmentaiton')

        self.parser.add_argument('--GaussianBlur', type=lambda x: bool(strtobool(x)), default='False',
                                 help='adding GaussianBlur in mri and ct in input images')
        self.parser.add_argument('--gaussian_sigma', type=float, default=1, help='gaussian_sigma')
        self.parser.add_argument('--gaussian_kernel_size', type=int, default=3, help='gaussian_kernel_size')

        self.parser.add_argument('--Local_Histogram_Equalization', type=lambda x: bool(strtobool(x)), default='False',
                                 help='Local_Histogram_Equalization in input images')
        self.parser.add_argument('--clip_limit', type=float, default=0.07, help='0.07')
        self.parser.add_argument('--LHE_kernel_size_x', type=int, default=50, help='LHE_kernel_size_x')
        self.parser.add_argument('--LHE_kernel_size_y', type=int, default=25, help='LHE_kernel_size_y')

        self.parser.add_argument('--Histogram_Equalization', type=lambda x: bool(strtobool(x)), default='False',
                                 help='adding Histogram_Equalization to CT')

        self.parser.add_argument('--B_normalization', type=lambda x: bool(strtobool(x)), default='True',
                                 help='adding CT normalization -1,1')
        self.parser.add_argument('--min_max_normalize', type=lambda x: bool(strtobool(x)), default='False',
                                 help='adding CT min max normalization ')


        self.parser.add_argument('--seg_norm', type=str, default='CrossEntropy',
                                 help='DiceNorm or CrossEntropy or CombinationLoss')
        self.parser.add_argument('--eval_step', type=bool, default=True, help='adding evaluation step')
        self.parser.add_argument('--eval_batch', type=int, default=42, help='adding evaluation step')

        self.parser.add_argument('--MC_uncertainty', type=lambda x: bool(strtobool(x)), default='True',
                                 help='adding monte carlo dropout uncertainty')
        self.parser.add_argument('--num_samples_uncertainty', type=int, default=10,
                                 help='number of images during test for calculating uncertainty')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization 0 or 1e-5')
        self.parser.add_argument('--weight_segmentation_in_GAN', type=float, default=1,
                                 help='weight of segmentation loss ')

        self.parser.add_argument('--segmentation_discriminator', type=bool, default=False,
                                 help='adding discriminator loss in segmentation')
        self.parser.add_argument('--just_segmentation', type=bool, default=False, help='just train segmentor in cyclegan')
        self.parser.add_argument('--separate_segmentation', type=bool, default=False,
                                 help='train translation and segmentation separately in cyclegan')
        self.parser.add_argument('--MRI_CT_segmentation', type=bool, default=False,
                                 help='train segmentaiton with mri and ct in cyclegan')

        self.parser.add_argument('--boundry_loss', type=bool, default=False,
                                 help='add boundry loss to the segmentation')
        self.parser.add_argument('--boundry_loss_weight', type=int, default=0.5,
                                 help='weight for binary loss segmentation ')
        self.parser.add_argument('--seg_rec_loss', type=bool, default=False,
                                 help='comparing segmentation loss of rec ct and real ct in cyclegan')
        self.parser.add_argument('--seg_fakeMRI_realCT_loss', type=bool, default=False,
                                 help='comparing segmentation loss of rec ct and real ct in cyclegan')
        self.parser.add_argument('--segmentation_linear_increase', type=bool, default=False,
                                 help='Increase linearly such that by epoch 25, your weight reaches 0.00001 in cyclegan')
        self.parser.add_argument('--segmentation_threshold_increase', type=bool, default=False,
                                 help='We keep the weight at 0 for the first 5 epochs and then increase it linearly to 0.00001 by epoch 25 in cyclegan')
        self.parser.add_argument('--perceptual_loss', type=bool, default=False, help=' adding perceptual loss in D_A')
        self.parser.add_argument('--direct_loss', type=bool, default=False, help=' adding direct loss in G_A and G_B i cyclegan')

        # self.parser.add_argument('--soft_real_fake', type=bool, default=False, help='')
        self.parser.add_argument('--Wasserstein_Lossy', type=bool, default=False, help='adding Wasserstein Loss')
        self.parser.add_argument('--segmentation_Discriminator', type=bool, default=False,
                                 help='add discriminator for segmentation model in cyclegan')

        self.parser.add_argument('--MIND_loss', type=bool, default=True, help='add MIND loss to the network ')
        self.parser.add_argument('--lambda_mind', type=float, default=0.1, help='weight for lambda_mind ')
        self.parser.add_argument('--MIND_sameModalityLoss', type=bool, default=False,
                                 help=' compare MRI with MRI and CT with CT')
        self.parser.add_argument('--MIND_diffModalityLoss', type=bool, default=True, help=' compare MRI with CT ')
        self.parser.add_argument('--MIND_sameModalityLossWeight', type=float, default=1,
                                 help='weight for same modality loss')
        self.parser.add_argument('--MIND_diffModalityLossWeight', type=float, default=1,
                                 help='weight for same modality loss')
        self.parser.add_argument('--non_local_region_size', type=int, default=9, help='non_local_region_size')
        self.parser.add_argument('--patch_size', type=int, default=7, help='patch_size')
        self.parser.add_argument('--neighbor_size', type=int, default=3, help='neighbor_size')
        self.parser.add_argument('--gaussian_patch_sigma', type=float, default=3.0, help='gaussian_patch_sigma')
        self.parser.add_argument('--MIND_loss_type', type=str, default='normal', help='normal or Cor_CoeLoss ')

        self.parser.add_argument('--lambda_cc', type=float, default=0.1, help='weight for lambda_cc ')
        self.parser.add_argument('--CC_sameModalityLoss', type=bool, default=False,
                                 help=' compare MRI with MRI and CT with CT')
        self.parser.add_argument('--CC_diffModalityLoss', type=bool, default=True, help=' compare MRI with CT ')

        self.parser.add_argument('--patience', type=int, default=5, help='patience in early stopping in cyclegan')
        self.parser.add_argument('--gradient_penalty', type=bool, default=True, help='adding gradient penalty')
        self.parser.add_argument('--gradient_penalty_lambda', type=int, default=2,
                                 help='represents the weight or strength of the gradient penalty regularization term')

        self.parser.add_argument('--patience_G', type=int, default=20, help='patience in early stopping')
        self.parser.add_argument('--patience_D', type=int, default=30, help='patience in early stopping')
        self.parser.add_argument('--patience_seg', type=int, default=20, help='patience in early stopping')
        self.parser.add_argument('--enable_early_stopping', type=bool, default=False, help='enable early stopping')
        self.parser.add_argument('--min_delta_G', type=float, default=0.0001,
                                 help='delta generator for early stopping ')
        self.parser.add_argument('--min_delta_D', type=float, default=0.0001,
                                 help='delta discriminator for early stopping ')
        self.parser.add_argument('--min_delta_seg', type=float, default=0.0001,
                                 help='delta seg for early stopping ')

        self.parser.add_argument('--leaky_relu', type=bool, default=False, help='adding leaky relu to the resnet')
        self.parser.add_argument('--D_G_ratio', type=int, default=3, help='adding leaky relu to the resnet')
        self.parser.add_argument('--uncertainty', type=bool, default=False, help='add dropout to the net')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./model_outputs/checkpoints/final2D',
                                 help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--yh_data_model', type=str, default='ImageWithMask', help='chooses the data location')
        self.parser.add_argument('--weight_2', type=int, default=1, help='weight of left kidney')
        self.parser.add_argument('--weight_3', type=int, default=1, help='weight of right kidney')
        self.parser.add_argument('--weight_7', type=int, default=1, help='weight of stomach')
        self.parser.add_argument('--test_CT_dir', type=str, default='./scratch/2D/CT', help='for test seg')
        self.parser.add_argument('--custom_sub_dir', type=int, default=0, help='custom_sub_dir')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            print('gpu', self.opt.gpu_ids)
            torch.cuda.set_device(self.opt.gpu_ids[0])
        else:
            print('cpu', self.opt.gpu_ids)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

