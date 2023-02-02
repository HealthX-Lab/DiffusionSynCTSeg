import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        # ------------------------- 01- main parameter ------------------------
        self.parser.add_argument('--operation', type=str, default='Train', help='chooses which Train, Test, TestSegMRI, TestMRI2CT, check_image')
        self.parser.add_argument('--input_type', type=dict, default={'Train':['MRI','CT','label'], 'Test':['CT','label'],  'TestSegMRI':['MRI','label'], 'TestMRI2CT':['MRI','CT'] })
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # ------------------------- 02- General Model ------------------------
        self.parser.add_argument('--model_type', type=str, default='GAN', help='Choose between GAN, ')
        # self.parser.add_argument('--model', type=str, default='cycle_gan',
        #                          help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter', type=int, default=100,
                                 help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--val_interval', type=int, default=1)
        # self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=64, help='then crop to this size')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--identity', type=float, default=0.0,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')

        # ------------------------- 03- GAN Model ------------------------
        # self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        # self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='SegResNet',
                                 help='selects model to use for netG between SegResNet, SegResNetVAE, UNet, SwinUNETR')
        self.parser.add_argument('--UNet_meatdata', type=dict,
                                 default={"spatial_dims":3,
                                        "in_channels":1,
                                        "out_channels":1,
                                        "channels":(16, 32, 64, 128, 256),
                                        "strides":(2,2,2,2),
                                        "kernel_size":3,
                                        "up_kernel_size":3,
                                        "num_res_units":0,
                                        "act":'PRELU',
                                        "norm":'INSTANCE',
                                        "dropout":0.0,
                                        "bias":True,
                                        "adn_ordering":'NDA',
                                        "dimensions":None,
                                          })
        self.parser.add_argument('--SegResNetVAE_meatdata', type=dict,
                                 default={"input_image_size":(64,64,64),
                                        "vae_estimate_std":False,
                                        "vae_default_std":0.3,
                                        "vae_nz":256,
                                        "spatial_dims":3,
                                        "init_filters":8,
                                        "in_channels":1,
                                        "out_channels":3,
                                        "dropout_prob":None,
                                        "act":('RELU', {'inplace': True}),
                                        # "norm":('GROUP', {'num_groups': 8}),
                                        "use_conv_final":True,
                                        "blocks_down":(1, 2, 2, 4),
                                        "blocks_up":(1, 1, 1),
                                        # "upsample_mode":UpsampleMode.NONTRAINABLE,
                                          })
        self.parser.add_argument('--SwinUNETR_meatdata', type=dict,
                                 default={"img_size":(64,64,64),
                                         "in_channels":1,
                                         "out_channels":1,
                                         "depths":(2, 2, 2, 2),
                                         "num_heads":(3, 6, 12, 24),
                                         "feature_size":24,
                                         "norm_name":'instance',
                                         "drop_rate":0.0,
                                         "attn_drop_rate":0.0,
                                         "dropout_path_rate":0.0,
                                         "normalize":True,
                                         "use_checkpoint":False,
                                         "spatial_dims":3,
                                         "downsample":"merging",
                                          })
        self.parser.add_argument('--SegResNet_meatdata', type=dict,
                                 default={"spatial_dims": 3,
                                          "init_filters": 8,
                                          "in_channels": 1,
                                          "out_channels": 2,
                                          "dropout_prob":0.2,
                                          # "norm": ('GROUP', {'num_groups': 8}),
                                          # "norm_name": '',
                                          # "num_groups": 8,
                                          "use_conv_final": True,
                                          "blocks_down": (1, 2, 2, 4),
                                          "blocks_up": (1, 1, 1),
                                          # "upsample_mode": UpsampleMode.NONTRAINABLE,
                                          })

        self.parser.add_argument('--Discriminator_basic_meatdata', type=dict,
                                 default={"in_shape": 3,
                                          "channels": 1,
                                          "strides": 2,
                                          "kernel_size": (16, 32, 64, 128, 256),
                                          "num_res_units ": (2, 2, 2, 2),
                                          "act": 'PRELU',
                                          "norm": 'INSTANCE',
                                          "dropout ": 0.2,
                                          "bias": True,
                                          "last_act": 'SIGMOID',
                                          })

        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        # self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        # ------------------------- 04- Seg Model ------------------------
        self.parser.add_argument('--which_model_netG_seg', type=str, default='SegResNet',
                                 help='selects model to use for netG between SegResNet, SegResNetVAE, UNet, SwinUNETR')
        self.parser.add_argument('--UNet_SEG_meatdata', type=dict,
                                 default={"spatial_dims": 3,
                                          "in_channels": 1,
                                          "out_channels": 3,
                                          "channels": (16, 32, 64, 128, 256),
                                          "strides": (2, 2, 2, 2),
                                          "kernel_size": 3,
                                          "up_kernel_size": 3,
                                          "num_res_units": 0,
                                          "act": 'PRELU',
                                          "norm": 'INSTANCE',
                                          "dropout": 0.0,
                                          "bias": True,
                                          "adn_ordering": 'NDA',
                                          "dimensions": None,
                                          })
        self.parser.add_argument('--SegResNetVAE_SEG_meatdata', type=dict,
                                 default={"input_image_size": (64, 64, 64),
                                          "vae_estimate_std": False,
                                          "vae_default_std": 0.3,
                                          "vae_nz": 256,
                                          "spatial_dims": 3,
                                          "init_filters": 8,
                                          "in_channels": 1,
                                          "out_channels": 3,
                                          "dropout_prob": None,
                                          "act": ('RELU', {'inplace': True}),
                                          # "norm": ('GROUP', {'num_groups': 8}),
                                          "use_conv_final": True,
                                          "blocks_down": (1, 2, 2, 4),
                                          "blocks_up": (1, 1, 1),
                                          # "upsample_mode": UpsampleMode.NONTRAINABLE,
                                          })
        self.parser.add_argument('--SwinUNETR_SEG_meatdata', type=dict,
                                 default={"img_size": (64, 64, 64),
                                          "in_channels": 1,
                                          "out_channels": 3,
                                          "depths": (2, 2, 2, 2),
                                          "num_heads": (3, 6, 12, 24),
                                          "feature_size": 24,
                                          "norm_name": 'instance',
                                          "drop_rate": 0.0,
                                          "attn_drop_rate": 0.0,
                                          "dropout_path_rate": 0.0,
                                          "normalize": True,
                                          "use_checkpoint": False,
                                          "spatial_dims": 3,
                                          "downsample": "merging",
                                          })
        self.parser.add_argument('--SegResNet_SEG_meatdata', type=dict,
                                 default={"spatial_dims": 3,
                                          "init_filters": 8,
                                          "in_channels": 1,
                                          "out_channels": 2,
                                          "dropout_prob": 0.2,
                                          # "norm": ('GROUP', {'num_groups': 8}),
                                          "norm_name": '',
                                          "num_groups": 8,
                                          "use_conv_final": True,
                                          "blocks_down": (1, 2, 2, 4),
                                          "blocks_up": (1, 1, 1),
                                          # "upsample_mode": UpsampleMode.NONTRAINABLE,
                                          })
        # ------------------------- 05- Dataset ------------------------
        self.parser.add_argument('--dataroot', type=str, default='/home/rtm/projects/rrg-eugenium/rtm/final_dataset')
        self.parser.add_argument('--val_rate', type=int, default=0.2)
        self.parser.add_argument('--attributes', type=dict,
                                 default={'Train':
                                                 {'NCCT':
                                                      {'folder_name': ['NCCT'],
                                                       'type': ['CT'],
                                                       'folder_paths': ['window_ct'],
                                                       'flag': True
                                                       },
                                                 'NeuroMorphometrics':
                                                       {'folder_name': ['Neuromorphometrics'],
                                                       'type': ['MRI', 'label'],
                                                       'folder_paths': ['with_Skull','ventricle/3ventricles'],
                                                       'flag': True
                                                       },
                                                 'OASIS3':
                                                      {'folder_name': ['OASIS3'],
                                                      'type': ['CT'],
                                                      'folder_paths': ['window_ct'],
                                                       'flag': True
                                                      }
                                               },
                                          'Test':
                                                {'iDB':
                                                     {'folder_name': ['iDB'],
                                                      'type': ['CT','label'],
                                                      'folder_paths': ['window_ct','beast_ventricle/3ventricles'],
                                                       'flag': True
                                                      }
                                                },
                                          'TestSegMRI':
                                              {'NeuroMorphometrics':
                                                       {'folder_name': ['Neuromorphometrics'],
                                                       'type': ['MRI', 'label'],
                                                       'folder_paths': ['with_Skull','ventricle/3ventricles'],
                                                       'flag': True
                                                       }
                                               },
                                          'TestMRI2CT':
                                              {'iDB':
                                                     {'folder_name': ['iDB'],
                                                      'type': ['MRI', 'CT'],
                                                      'folder_paths': ['window_ct','with_Skull'],
                                                       'flag': True
                                                      }
                                              }
                                          })

        # ------------------------- 05- Data Augmentation ------------------------
        self.parser.add_argument('--oversample', type=bool, default=0)
        self.parser.add_argument('--cache_rate', type=float, default=1.0)
        self.parser.add_argument('--dataLoader_num_workers', type=int, default=1)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--transformers', type=dict,
                                 default= {'Train':
                                                [
                                                {'cls': 'LoadImaged',
                                                 'arg': {'keys': ['CT','MRI','label']}},
                                                {'cls': 'EnsureChannelFirstd',
                                                 'arg': {'keys': ['CT','MRI','label']}},
                                                {'cls': 'RandCropByLabelClassesd',
                                                 'arg': {'keys': ['CT','MRI','label'],
                                                        'label_key': "label",
                                                        'spatial_size': (64, 64, 64),
                                                        'num_samples': 1,
                                                         "num_classes":3,
                                                        }
                                                 },
                                                {'cls': 'AsDiscreted',
                                                 'arg': {'keys': ['label'],
                                                         'to_onehot':3,
                                                         }
                                                }
                                                ],
                                           'val':
                                               [
                                                   {'cls': 'LoadImaged',
                                                    'arg': {'keys': ['CT', 'MRI', 'label']}},
                                                   {'cls': 'EnsureChannelFirstd',
                                                    'arg': {'keys': ['CT', 'MRI', 'label']}},
                                                   # {'cls': 'RandCropByLabelClassesd',
                                                   #  'arg': {'keys': ['CT', 'MRI', 'label'],
                                                   #          'label_key': "label",
                                                   #          'spatial_size': (64, 64, 64),
                                                   #          'num_samples': 1,
                                                   #          "num_classes": 3,
                                                   #          }
                                                   #  },
                                                   # {'cls': 'AsDiscreted',
                                                   #  'arg': {'keys': ['label'],
                                                   #          'to_onehot': 3,
                                                   #          }
                                                   #  }
                                               ],
                                          'Test':
                                                [
                                                  {'cls': 'LoadImaged',
                                                   'arg': {'keys': ['CT', 'MRI', 'label']}},
                                                  {'cls': 'EnsureChannelFirstd',
                                                   'arg': {'keys': ['CT', 'MRI', 'label']}},
                                                  {'cls': 'RandCropByLabelClassesd',
                                                   'arg': {'keys': ['CT', 'MRI', 'label'],
                                                           'label_key': "label",
                                                           'spatial_size': (64, 64, 64),
                                                           'num_samples': 1,
                                                           }
                                                   },
                                                  {'cls': 'AsDiscreted',
                                                   'arg': {'keys': ['label'],
                                                           'to_onehot': 3,
                                                           }
                                                   }
                                              ],
                                         })

        # ------------------------- 06- Login  ------------------------
        self.parser.add_argument('--val_template_save_path', type=str, default='./save_images/val_template.png')
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

    def parse(self):

        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        #
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