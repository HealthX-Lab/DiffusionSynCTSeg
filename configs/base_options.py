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

        # ------------------------- 03- Cycle seg model------------------------
        self.parser.add_argument('--continue_train', type=bool, default=True, help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=int, default=1, help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--num_classes', type=int, default=5, help='# of output classes')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002,
                                 help='initial learning rate for adam')  # ************************************************0.9
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--identity', type=float, default=0.0,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--lr_enable', type=bool, default=False, help='enabeling lr flag')
        self.parser.add_argument('--lr_scheduler_GAN', type=dict,
                                 default={'cls': 'StepLR', 'arg': {'step_size': 30, 'gamma': 0.1}})
        self.parser.add_argument('--lr_scheduler_discriminator', type=dict,
                                 default={'cls': 'StepLR', 'arg': {'step_size': 30, 'gamma': 0.1}})
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')

        # ------------------------- 04- Training Chain ------------------------
        self.parser.add_argument('--model_type', type=str, default='GAN', help='Choose between GAN, ')
        self.parser.add_argument('--max_epochs', type=int, default=4, help='# of epochs ')
        self.parser.add_argument('--ConfusionMatrixMetric', type=tuple, default=(
            "sensitivity", "precision", "recall", 'specificity'), help='Confusion Matrix Metric')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--val_interval', type=int, default=1)
        self.parser.add_argument('--cache_rate', type=float, default=1.0)
        self.parser.add_argument('--dataLoader_num_workers', type=int, default=1)
        self.parser.add_argument('--batch_size', type=int, default=1)


        # ------------------------- 05- Base Model ------------------------
        self.parser.add_argument('--seg_norm', type=str, default='DiceNorm', help='DiceNorm or CrossEntropy')
        self.parser.add_argument('--cross_entropy_weight', type=list, default=[1,1,1,1,1], help='cross entropy weights for segmentation')
        self.parser.add_argument('--VAL_AMP', type=bool, default=False, help='amp. MONAI has exposed this feature in the workflow implementations by providing access to the amp parameter')

        # ------------------------- 05- Network ------------------------
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='UNet',
                                 help='selects model to use for netG between SegResNet, SegResNetVAE, UNet, SwinUNETR')
        self.parser.add_argument('--UNet_meatdata', type=dict,
                                 default={"spatial_dims":3,
                                        "in_channels":1,
                                        "out_channels":1,
                                        "channels":(16, 32, 64,128),
                                        "strides":(2,2,2),
                                        "kernel_size":3,
                                        "up_kernel_size":3,
                                        "num_res_units":0,
                                        "act":'PRELU',
                                        # "norm":'INSTANCE',
                                        # "dropout":0.0,
                                        # "bias":True,
                                        # "adn_ordering":'NDA',
                                        # "dimensions":None,
                                          })
        self.parser.add_argument('--SegResNetVAE_meatdata', type=dict,
                                 default={"input_image_size":(64,64,64),
                                        "vae_estimate_std":False,
                                        "vae_default_std":0.3,
                                        "vae_nz":256,
                                        "spatial_dims":3,
                                        "init_filters":8,
                                        "in_channels":1,
                                        "out_channels":1,
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
                                          "out_channels": 1,
                                          "dropout_prob":0.2,
                                          # "norm": ('GROUP', {'num_groups': 8}),
                                          # "norm_name": '',
                                          # "num_groups": 8,
                                          # "use_conv_final": True,
                                          # "blocks_down": (1, 2, 2, 4),
                                          # "blocks_up": (1, 1, 1),
                                          # "upsample_mode": UpsampleMode.NONTRAINABLE,
                                          })

        self.parser.add_argument('--Discriminator_basic_metadata', type=dict,
                                 default={"in_shape": (1,64,64,64),
                                          "channels": (8, 16, 32),#, 64
                                          "strides": (2, 2, 2 ),
                                          "kernel_size": 5,
                                          "num_res_units": 1,
                                          "act": 'PRELU',
                                          "norm": 'INSTANCE',
                                          "dropout": 0.2,
                                          "bias": True,
                                          "last_act": 'SIGMOID',
                                          })

        self.parser.add_argument('--Gen_dropout', type=float, default=0.2,
                                 help='# drop out in last layer in generator model')

        self.parser.add_argument('--which_model_netG_seg', type=str, default='SegResNet',
                                 help='selects model to use for netG between SegResNet, SegResNetVAE, UNet, SwinUNETR')
        self.parser.add_argument('--UNet_SEG_meatdata', type=dict,
                                 default={"spatial_dims": 3,
                                          "in_channels": 1,
                                          "out_channels": 5,
                                          "channels": (16, 32, 64),
                                          "strides": (2, 2),
                                          "kernel_size": 3,
                                          "up_kernel_size": 3,
                                          "num_res_units": 0,
                                          "act": 'PRELU',
                                          # "norm": 'INSTANCE',
                                          # "dropout": 0.0,
                                          # "bias": True,
                                          # "adn_ordering": 'NDA',
                                          # "dimensions": None,
                                          })
        self.parser.add_argument('--SegResNetVAE_SEG_meatdata', type=dict,
                                 default={"input_image_size": (64, 64, 64),
                                          "vae_estimate_std": False,
                                          "vae_default_std": 0.3,
                                          "vae_nz": 256,
                                          "spatial_dims": 3,
                                          "init_filters": 8,
                                          "in_channels": 1,
                                          "out_channels": 5,
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
                                          "out_channels": 5,
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
                                          "out_channels": 5,
                                          "dropout_prob": 0.2,
                                          # "norm": ('GROUP', {'num_groups': 8}),
                                          "norm_name": '',
                                          "num_groups": 8,
                                          "use_conv_final": True,
                                          "blocks_down": (1, 2, 2, 4),
                                          "blocks_up": (1, 1, 1),
                                          # "upsample_mode": UpsampleMode.NONTRAINABLE,
                                          })

        # ------------------------- 05- Dataset Collection ------------------------
        self.parser.add_argument('--input_type', type=dict, default={'Train':['MRI','CT','label'], 'Test':['CT','label'],  'TestSegMRI':['MRI','label'], 'TestMRI2CT':['MRI','CT'] })
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

        self.parser.add_argument('--oversample', type=bool, default=0)
        # ------------------------- 06- Test  ------------------------
        self.parser.add_argument('--roi_size', type = tuple, default=(64,64,64))
        self.parser.add_argument('--sw_batch_size', type=int, default=1)
        # ------------------------- 06- Transformer  ------------------------
        self.parser.add_argument('--transformers', type=dict,
                                 default= {'Train':
                                                [
                                                {'cls': 'LoadImaged',
                                                 'arg': {'keys': ['CT','MRI','label']}},
                                                {'cls': 'EnsureChannelFirstd',
                                                 'arg': {'keys': ['CT','MRI','label']}},
                                                {'cls': 'ConvertToMultiChannelVentricleClasses',
                                                 'arg': {'keys': ['label']}},
                                                {'cls': 'RandCropByLabelClassesd',
                                                 'arg': {'keys': ['CT','MRI','label'],
                                                        'label_key': "label",
                                                        'spatial_size': [64,64,64],
                                                        'num_samples': 1,
                                                         # 'ratios':[1, 1, 1, 1, 1],
                                                         "num_classes":5,
                                                        }
                                                 },
                                                ],
                                           'val':
                                               [
                                                   {'cls': 'LoadImaged',
                                                    'arg': {'keys': ['CT', 'MRI', 'label']}},
                                                   {'cls': 'EnsureChannelFirstd',
                                                    'arg': {'keys': ['CT', 'MRI', 'label']}},
                                                   {'cls': 'ConvertToMultiChannelVentricleClasses',
                                                    'arg': {'keys': ['label']}},
                                                   # {'cls': 'RandCropByLabelClassesd',
                                                   #  'arg': {'keys': ['CT', 'MRI', 'label'],
                                                   #          'label_key': "label",
                                                   #          'spatial_size': [64,64,64],
                                                   #          'num_samples': 1,
                                                   #          # 'ratios': [1, 1, 1, 1, 1],
                                                   #          "num_classes": 5,
                                                   #          }
                                                   #  },

                                               ],
                                          'Test':
                                                [
                                                  {'cls': 'LoadImaged',
                                                   'arg': {'keys': ['CT', 'label']}},
                                                    {'cls': 'ConvertToMultiChannelVentricleClasses',
                                                     'arg': {'keys': ['label']}},
                                                  {'cls': 'EnsureChannelFirstd',
                                                   'arg': {'keys': ['CT', 'label']}},
                                                  # {'cls': 'RandCropByLabelClassesd',
                                                  #  'arg': {'keys': ['CT', 'MRI', 'label'],
                                                  #          'label_key': "label",
                                                  #          'spatial_size': (64,64,64),
                                                  #          'num_samples': 1,
                                                  #          "num_classes": 5,
                                                  #          }
                                                  #  },

                                              ],
                                         })

        # ------------------------- 06- Log  ------------------------
        self.parser.add_argument('--val_template_save_path', type=str, default='./save_images/val_template.png')
        self.parser.add_argument('--name', type=str, default='first',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
        self.parser.add_argument('--logs_dir', type=str, default='logs',
                                 help='models are saved here')
        self.parser.add_argument('--test_image_dir', type=str, default='test_images',
                                 help='test images are saved here')
        self.parser.add_argument('--val_iamge_dir', type=str, default='val_images',
                                 help='val images are saved here')
        self.parser.add_argument('--metrics_image_dir', type=str, default='metrics_images',
                                 help='metricsimages are saved here')
        self.parser.add_argument('--save_slices', type=list, default=[45,70,100],
                                 help='image slices that are going to save')

    def parse(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parser.add_argument('--device',  default=device)

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


