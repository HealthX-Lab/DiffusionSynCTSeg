import time
import os
import sublist
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torch
import pandas as pd
import numpy as np
from util.util import append_data_to_csv

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


opt = TrainOptions().parse()

# Method = 'ImageOnly'
Method = opt.yh_data_model

# paired models
# raw_MRI_dir = '/home/rtm/scratch/2D/MRI'
# raw_MRI_seg_dir = '/home/rtm/scratch/2D/seg'
# raw_CT_dir = '/home/rtm/scratch/2D/clip_min_max_preprocess/iDB'
# sub_list_dir = '/home/rtm/scratch/2D/clip_min_max_preprocess/sublists'


# unpaired models
raw_MRI_dir = '/home/rtm/scratch/2D/MRI'
raw_MRI_seg_dir = '/home/rtm/scratch/2D/seg'
raw_CT_dir = '/home/rtm/scratch/2D/clip_min_max_preprocess'#'/home/rtm/scratch/2D/clip_min10_max200_std_preprocess/iDB'#'/home/rtm/scratch/2D/clip_min_max_preprocess/iDB'
sub_list_dir = '/home/rtm/scratch/2D/clip_m10_500_std_preprocess/sublists'


# raw_MRI_dir = '/home/rtm/scratch/2D/MRI'
# raw_MRI_seg_dir = '/home/rtm/scratch/2D/seg'
# raw_CT_dir = '/home/rtm/scratch/2D/CT'
# sub_list_dir = '/home/rtm/scratch/2D/sublists/sublist2'

# raw_MRI_dir = '/home/rtm/scratch/2D/MRI'
# raw_MRI_seg_dir = '/home/rtm/scratch/2D/seg'
# raw_CT_dir = '/home/rtm/scratch/2D/mixed_CT_preOASIS_NCCT'
# sub_list_dir = '/home/rtm/scratch/2D/sublists/seperate_sublists'


ngpus_per_node = torch.cuda.device_count()
local_rank = int(os.environ.get("SLURM_LOCALID"))
rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
print('******* rank  ', rank,local_rank,flush= True)
opt.gpu_ids = [local_rank]
print('***** gpu ids', opt.gpu_ids , flush=True)
#
# torch.cuda.set_device(opt.gpu_ids)


TrainOrTest = opt.yh_run_model #'Train' #
df = pd.DataFrame(columns = ['D_A','G_A','Cyc_A','D_B', 'G_B','Cyc_B','Seg'])
val_df = pd.DataFrame(columns = ['D_A','G_A','Cyc_A','D_B', 'G_B','Cyc_B','Seg'])





#evaluation
if TrainOrTest == 'Test':
    print('in test***************')

    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.isTrain = False
    opt.phase = 'test'
    opt.no_dropout = True

    sub_list_MRI = ''
    sub_list_CT = ''
    sub_list_seg = ''


    # if not os.path.exists(cycle_output_dir):
    #     cycle_output_dir = '/scratch/huoy1/projects/DeepLearning/Cycle_Deep/Output/CycleTest'


    # mkdir(sub_list_dir)
    # sub_list_MRI = os.path.join(sub_list_dir, 'test_MRI_iDB_paired.txt')
    # sub_list_CT = os.path.join(sub_list_dir, 'test_CT_iDB_paired.txt')  # test_CT_iDB_paired_m10_200.txt   test_CT_iDB_paired.txt
    # sub_list_seg = os.path.join(sub_list_dir, 'test_seg_iDB_paired.txt')

    # sub_list_MRI = os.path.join(sub_list_dir, 'test_MRI_list.txt')
    # sub_list_CT = os.path.join(sub_list_dir, 'test_CT_list.txt')
    # sub_list_seg = os.path.join(sub_list_dir, 'test_seg_sublist.txt')

    # #iDB test data
    if opt.test_dataset_iDB_10_500_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')



    # train unpaired -10 500
    # elif opt.test_dataset_train_10_500_80_120:
        # sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
        # sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_80_120.txt')#oasis_preprocess_80_120.txt or oasis_ncct_80_120.txt
        # sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_80_120.txt')


    #
    # elif opt.test_dataset_iDB_10_500_85_100:
        # sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_85_100.txt')
        # sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_85_100.txt')
        # sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_85_100.txt')

    # elif opt.test_dataset_iDB_10_400_80_120:
        # sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        # sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_400_80_120.txt')
        # sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    # elif opt.test_dataset_iDB_10_300_80_120:
        # sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        # sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_300_80_120.txt')
        # sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    # elif opt.test_dataset_iDB_10_200_80_120:
        # sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        # sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_200_80_120.txt')
        # sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    # elif opt.test_dataset_iDB_10_100_80_120:
        # sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        # sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_100_80_120.txt')
        # sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    # elif opt.test_dataset_iDB_10_50_80_120:
        # sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        # sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_50_80_120.txt')
        # sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')




    imglist_MRI = sublist.dir2list(raw_MRI_dir, sub_list_MRI)
    imglist_CT = sublist.dir2list(raw_CT_dir, sub_list_CT)
    imglist_seg = sublist.dir2list(raw_MRI_seg_dir, sub_list_seg)

    imglist_MRI, imglist_CT,imglist_seg = sublist.equal_length_two_list(imglist_MRI, imglist_CT,imglist_seg);

    # input the opt that we want
    opt.raw_MRI_dir = raw_MRI_dir
    opt.raw_MRI_seg_dir = raw_MRI_seg_dir
    opt.raw_CT_dir = raw_CT_dir
    opt.imglist_MRI = imglist_MRI
    opt.imglist_CT = imglist_CT
    opt.imglist_seg = imglist_seg



    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print('shape dataset *********888',np.shape(dataset))
    dataset_size = len(data_loader)


    models_name = [


                      ]
    model_epochs = [

                    ]

    segmentation_model = [

                    ]


    for index in range(len(models_name)):
        opt.which_model_netSeg = segmentation_model[index]
        df = pd.DataFrame()
        opt.name = models_name[index]
        max_epoch = model_epochs[index]
        print('*&*&*&*^^^ ',opt.name, max_epoch)
        for epoch_number in range(0,max_epoch+1):#max_epoch+1
            opt.which_epoch = epoch_number
            opt.test_seg_output_dir = f'/home/rtm/scratch/model_outputs/Output_save/2D/cyclegan/{opt.name}/iBD_test2/epoch_{epoch_number}'
            cycle_output_dir = opt.test_seg_output_dir
            filename = f'/home/rtm/scratch/model_outputs/csvfiles/2D/innovatives2/cyclegan/{opt.name}_iDB_test.csv'
            print(filename,'**',cycle_output_dir)
            print('#testing images = %d' % dataset_size)
            model = create_model(opt)
            visualizer = Visualizer(opt)
            for i, data in enumerate(dataset):
                # model = create_model(opt)
                # visualizer = Visualizer(opt)
                model.set_zero()
                model.set_input(data)
                model.test()
                # print('process image... %s' % img_path)
                if opt.MC_uncertainty :
                    # visuals = model.get_current_visuals()
                    # image_paths_A, image_paths_B, image_paths_seg= model.get_image_paths()
                    print('*** $$$$ @@@',epoch_number,' * ',i)
                    coef = model.get_coef()
                    # print(coef)
                    df_test = pd.DataFrame([coef])
                    df_test['epoch'] = epoch_number
                    df_test['data_number'] = i
                    # df = pd.concat([df, df_test], ignore_index=True)
                    # visualizer.save_images_to_dir_uncertainty(cycle_output_dir, visuals, img_path)
                    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
                        df_test.to_csv(filename, index=False)
                    else:
                        df_test.to_csv(filename, mode='a', header=False, index=False)


                elif not opt.MC_uncertainty:
                    # print('*&*&*&*%')
                    visuals = model.get_current_visuals()
                    image_paths_A, image_paths_B, image_paths_seg = model.get_image_paths()
                    visualizer.save_images_to_dir(cycle_output_dir, visuals, image_paths_A, image_paths_B, image_paths_seg)
            # if opt.MC_uncertainty:
            #     coef = model.get_coef()
            #     df_test = pd.DataFrame([coef])
            #     df_test['epoch'] = epoch_number
            #     df = pd.concat([df, df_test], ignore_index=True)
        # df.to_csv(f'/home/rtm/scratch/model_outputs/csvfiles/2D/{opt.name}_add_iou_.csv', index=False)

elif TrainOrTest == 'TestSeg':
    print('in test***************')

    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.isTrain = False
    opt.phase = 'test'
    opt.no_dropout = True
    sub_list_MRI = ''
    sub_list_CT = ''
    sub_list_seg = ''

    # if not os.path.exists(cycle_output_dir):
    #     cycle_output_dir = '/scratch/huoy1/projects/DeepLearning/Cycle_Deep/Output/CycleTest'

    # mkdir(sub_list_dir)
    # sub_list_MRI = os.path.join(sub_list_dir, 'test_MRI_iDB_paired.txt')
    # sub_list_CT = os.path.join(sub_list_dir, 'test_CT_iDB_paired.txt')  # test_CT_iDB_paired_m10_200.txt   test_CT_iDB_paired.txt
    # sub_list_seg = os.path.join(sub_list_dir, 'test_seg_iDB_paired.txt')

    # sub_list_MRI = os.path.join(sub_list_dir, 'test_MRI_list.txt')
    # sub_list_CT = os.path.join(sub_list_dir, 'test_CT_list.txt')
    # sub_list_seg = os.path.join(sub_list_dir, 'test_seg_sublist.txt')

    # #iDB test data
    if opt.test_dataset_iDB_10_500_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    # train unpaired -10 500
    elif opt.test_dataset_train_10_500_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_80_120.txt')#oasis_preprocess_80_120.txt or oasis_ncct_80_120.txt
        sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_80_120.txt')


    elif opt.test_dataset_iDB_10_500_85_100:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_85_100.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_85_100.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_85_100.txt')

    elif opt.test_dataset_iDB_10_400_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_400_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    elif opt.test_dataset_iDB_10_300_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_300_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    elif opt.test_dataset_iDB_10_200_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_200_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    elif opt.test_dataset_iDB_10_100_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_100_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    elif opt.test_dataset_iDB_10_50_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_m10_50_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    imglist_MRI = sublist.dir2list(raw_MRI_dir, sub_list_MRI)
    imglist_CT = sublist.dir2list(raw_CT_dir, sub_list_CT)
    imglist_seg = sublist.dir2list(raw_MRI_seg_dir, sub_list_seg)

    imglist_MRI, imglist_CT, imglist_seg = sublist.equal_length_two_list(imglist_MRI, imglist_CT, imglist_seg);

    # input the opt that we want
    opt.raw_MRI_dir = raw_MRI_dir
    opt.raw_MRI_seg_dir = raw_MRI_seg_dir
    opt.raw_CT_dir = raw_CT_dir
    opt.imglist_MRI = imglist_MRI
    opt.imglist_CT = imglist_CT
    opt.imglist_seg = imglist_seg

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print('shape dataset *********888', np.shape(dataset))
    dataset_size = len(data_loader)

    models_name = [

    ]
    model_epochs = [


    ]



    for index in range(len(models_name)):
        # opt.which_model_netSeg = segmentation_model[index]
        df = pd.DataFrame()
        opt.name = models_name[index]
        max_epoch = model_epochs[index]
        print('*&*&*&*^^^ ', opt.name, max_epoch)
        for epoch_number in range(0, max_epoch + 1):  # max_epoch+1
            opt.which_epoch = epoch_number
            opt.test_seg_output_dir = f'/home/rtm/scratch/model_outputs/Output_save/2D/cyclegan/{opt.name}/iBD_test2/epoch_{epoch_number}'
            cycle_output_dir = opt.test_seg_output_dir
            filename = f'/home/rtm/scratch/model_outputs/csvfiles/2D/innovatives2/cyclegan/{opt.name}_iDB_test.csv'
            print(filename, '**', cycle_output_dir)
            print('#testing images = %d' % dataset_size)
            model = create_model(opt)
            visualizer = Visualizer(opt)
            for i, data in enumerate(dataset):
                # model = create_model(opt)
                # visualizer = Visualizer(opt)
                model.set_zero()
                model.set_input(data)
                model.test()
                # print('process image... %s' % img_path)
                if opt.MC_uncertainty:
                    # visuals = model.get_current_visuals()
                    # image_paths_A, image_paths_B, image_paths_seg= model.get_image_paths()
                    print('*** $$$$ @@@', epoch_number, ' * ', i)
                    coef = model.get_coef()
                    # print(coef)
                    df_test = pd.DataFrame([coef])
                    df_test['epoch'] = epoch_number
                    df_test['data_number'] = i
                    # df = pd.concat([df, df_test], ignore_index=True)
                    # visualizer.save_images_to_dir_uncertainty(cycle_output_dir, visuals, img_path)
                    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
                        df_test.to_csv(filename, index=False)
                    else:
                        df_test.to_csv(filename, mode='a', header=False, index=False)


                elif not opt.MC_uncertainty:
                    # print('*&*&*&*%')
                    visuals = model.get_current_visuals()
                    image_paths_A, image_paths_B= model.get_image_paths()
                    visualizer.save_cycle_gan_images_to_dir(cycle_output_dir, visuals, image_paths_A, image_paths_B)
            # if opt.MC_uncertainty:
            #     coef = model.get_coef()
            #     df_test = pd.DataFrame([coef])
            #     df_test['epoch'] = epoch_number
            #     df = pd.concat([df, df_test], ignore_index=True)
        # df.to_csv(f'/home/rtm/scratch/model_outputs/csvfiles/2D/{opt.name}_add_iou_.csv', index=False)

elif TrainOrTest == 'Train':
    print('in the train')
    sub_list_MRI = ''
    sub_list_CT = ''
    sub_list_seg = ''

    # mkdir(sub_list_dir)
    # paired
    if opt.train_dataset_paired_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_preprocess_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    elif opt.normalize_dataset_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'combine_MRI_IXI_Neuto_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'normalized_ncct_oasis_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'combine_label_MRI_IXI_Neuto_80_120.txt')

    # combine data IXI and Neuromorthometrics
    elif opt.train_dataset_IXI_neuro_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'combine_MRI_IXI_Neuto_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'combine_label_MRI_IXI_Neuto_80_120.txt')

    # # unpaired -10 500
    elif opt.train_dataset_neuro_80_120:
        sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_80_120.txt')#oasis_preprocess_80_120.txt or oasis_ncct_80_120.txt
        sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_80_120.txt')

    # # unpaired -10 500 all labels
    # elif opt.train_dataset_IXI_neuro_80_120:
    #     sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
    #     sub_list_CT = os.path.join(sub_list_dir,
    #                                'oasis_ncct_80_120.txt')  # oasis_preprocess_80_120.txt or oasis_ncct_80_120.txt
    #     sub_list_seg = os.path.join(sub_list_dir, 'neuro_seg_all_labels_expand_80_120.txt')

    # # unpaired -10 500 all slices
    elif opt.train_dataset_neuro_all_slices:
        sub_list_MRI = os.path.join(sub_list_dir, 'sublist_all_mri_train.txt')
        sub_list_CT = os.path.join(sub_list_dir,'sublist_all_CT_train.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'sublist_all_seg_train.txt')

    # paired segmentation
    elif opt.train_dataset_finetune:
        sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'real_fake_CT_finetune_segmentation.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'real_fake_CT_label_finetune_segmentation.txt')


    # 0.6 best images in var for segmentation
    elif opt.train_dataset_finetune_06:
        sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'priority_CT_new_file_0.6.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'priority_label_new_file_0.6.txt')

    elif opt.train_dataset_neuro_85_100:
        sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_85_100.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_85_100.txt')  # oasis_preprocess_80_120.txt or oasis_ncct_80_120.txt
        sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_85_100.txt')

    elif opt.train_dataset_IXI_neuro_all_slice:
        sub_list_MRI = os.path.join(sub_list_dir, 'all_slices_MRI_IXI_preprocess.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'sublist_all_CT_train.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'all_slices_MRI_label_IXI_preprocess.txt')


    opt.raw_MRI_dir = raw_MRI_dir
    opt.raw_MRI_seg_dir = raw_MRI_seg_dir
    opt.raw_CT_dir = raw_CT_dir
    if opt.eval_step:
        sub_list_MRI_val = os.path.join(sub_list_dir, 'normalize_iDB_MRI_80_120.txt')
        sub_list_CT_val = os.path.join(sub_list_dir, 'normalize_iDB_CT_80_120.txt')
        sub_list_seg_val = os.path.join(sub_list_dir, 'normalize_iDB_seg_80_120.txt')

        imglist_MRI_val = sublist.dir2list(raw_MRI_dir, sub_list_MRI_val)
        imglist_CT_val = sublist.dir2list(raw_CT_dir, sub_list_CT_val)
        imglist_seg_val = sublist.dir2list(raw_MRI_seg_dir, sub_list_seg_val)

        imglist_MRI_val = imglist_MRI_val[0:21] + imglist_MRI_val[41:62]#[9:11]
        imglist_CT_val = imglist_CT_val[0:21] + imglist_CT_val[41:62]#[9:11]
        imglist_seg_val = imglist_seg_val[0:21] + imglist_seg_val[41:62]#[9:11]

        opt.imglist_MRI_val = imglist_MRI_val
        opt.imglist_CT_val = imglist_CT_val
        opt.imglist_seg_val = imglist_seg_val
        from data.val_data_loader import valSegDataset
        val_data_obj = valSegDataset()
        val_data_obj.initialize(opt)
        val_dataset = torch.utils.data.DataLoader(
            val_data_obj,
            batch_size=opt.eval_batch,
            num_workers=int(opt.nThreads))
        # val_dataset = val_dataloader.load_data()
        val_dataset_size = len(val_data_obj)


    # sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri.txt')
    # sub_list_CT = os.path.join(sub_list_dir, 'sublist_CT.txt')
    # sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg.txt')

    # sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
    # sub_list_CT = os.path.join(sub_list_dir, 'ncct_pre_oasis_80_120.txt')
    # sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_80_120.txt')

    imglist_MRI = sublist.dir2list(raw_MRI_dir, sub_list_MRI)
    imglist_CT = sublist.dir2list(raw_CT_dir, sub_list_CT)
    imglist_seg = sublist.dir2list(raw_MRI_seg_dir, sub_list_seg)

    imglist_MRI, imglist_CT,imglist_seg = sublist.equal_length_two_list(imglist_MRI, imglist_CT,imglist_seg);
    print('after sublist')
    # input the opt that we want

    opt.imglist_MRI = imglist_MRI
    opt.imglist_CT = imglist_CT
    opt.imglist_seg = imglist_seg

    # for i , j in zip(imglist_MRI,imglist_CT):
    #     print('** ',i,' ** ',j)
    opt.crossentropy_weight = [1,30]#,10,10,1,10,1
    print('after data leader')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    print('#model created')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        print('lr: ',opt.lr)
        if opt.segmentation_linear_increase:
            opt.weight_segmentation_in_GAN = 0.00001 * epoch / 25
        elif opt.segmentation_threshold_increase:
            if epoch<5:
                opt.weight_segmentation_in_GAN = 0
            else:
                opt.weight_segmentation_in_GAN = 0.00001 * (epoch - 4)/21
        print('opt.weight_segmentation_in_GAN : ',opt.weight_segmentation_in_GAN)




        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data,i)
            # print('*****before optimization*****')
            model.optimize_parameters(i,epoch)
            # print('*****after optimization*****')

            if total_steps % opt.print_freq == 0:
                print('total_steps: ',total_steps)
                errors = model.get_current_errors()
                csv_file_path = f'{opt.test_seg_output_dir}.csv'
                append_data_to_csv(csv_file_path, errors)
                # print(total_steps,errors)
                new_df = pd.DataFrame([errors])
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(f'/home/rtm/scratch/model_outputs/csvfiles/2D/save_best/{opt.name}.csv', index=False)
                t = (time.time() - iter_start_time) / opt.batchSize

                if opt.eval_step and total_steps % (opt.print_freq) == 0:
                    for val_num, val_data in enumerate(val_dataset):
                        model.test(val_data)
                        val_result = model.get_val_result()
                        # val_images = model.get_val_images()
                        model.save_best_model()
                        val_csv_file_path = f'{opt.test_seg_output_dir}_val.csv'
                        append_data_to_csv(val_csv_file_path, val_result)
                        new_val_df = pd.DataFrame([val_result])
                        val_df = pd.concat([val_df, new_val_df], ignore_index=True)
                        val_df.to_csv(f'/home/rtm/scratch/model_outputs/csvfiles/2D/save_best/{opt.name}_validation.csv',
                                  index=False)
                        # visualizer.save_val_images_to_dir(f'{opt.test_seg_output_dir}/validation_images2', val_images,total_steps,val_num)






            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if opt.enable_early_stopping == 1:
            if model.stop_training == 1:
                print('saving the latest early stopping model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save(f'latest_early_stopping_{epoch}')
                break

            model.earlyStopping(epoch)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()


