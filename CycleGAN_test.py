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
from distutils.util import strtobool
from util.util import save_image_array

def mkdir(path):
    print('in mkdir ',path)
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"An error occurred while creating directories: {e}")


print('hellooooo')
opt = TrainOptions().parse()

# Method = 'ImageOnly'
Method = opt.yh_data_model


params = [
{

    'name': 'zero_centered_MIND_CC_cyclegan_segmentation',
    'which_model_netG': 'resnet_9blocks',
    'which_model_netSeg': 'R2AttU_Net',

    'model': 'test',
    'test_dataset_iDB_10_500_80_120': 'True',
    'test_dataset_train_10_500_80_120': 'False',
    'test_dataset_iDB_10_500_85_100': 'False',
    'test_dataset_iDB_10_400_80_120': 'False',
    'test_dataset_iDB_10_300_80_120': 'False',
    'test_dataset_iDB_10_200_80_120': 'False',
    'test_dataset_iDB_10_100_80_120': 'False',
    'test_dataset_iDB_10_50_80_120': 'False',
     'test_iDB_normalize_dataset_10_500_80_120': 'False',

    'GaussianBlur': 'False',
    'Local_Histogram_Equalization': 'False',
    'Histogram_Equalization': 'False',
    'B_normalization': 'True',
    'min_max_normalize': 'False',
    'yh_run_model': 'Test',
    'MC_uncertainty': 'True',
    'num_samples_uncertainty': 10,
    'max_epoch': 200,
    'print_images_with_uncertainty': 'False',
    'len_datase': 0,
    'folder_name': 'cyclegan_30',
    'attribute': '3D_dice',
    'path_csv': '/home/rtm/scratch/model_outputs/csvfiles/final_with_histogram',
    'path_images': '/home/rtm/scratch/model_outputs/Output_save/final_with_histogram',

},
]

for param_dict in params:

    for key , val in param_dict.items():
        if val in ['True', 'False']:  # if the value is a string 'True'/'False'
            val = bool(strtobool(val))
        setattr(opt, key, val)

    args = vars(opt)
    opt.eval = True


    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    raw_MRI_dir = 'path-to-2D-MRI test dataset'
    raw_MRI_seg_dir = 'path-to-2D-label test dataset'
    raw_CT_dir = 'path-to-2D-CT test dataset'
    sub_list_dir = 'path-to-2D-sublists test dataset'  # txt files that contain name of all 2D images with their paths




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
    # opt.path_images = '/home/rtm/projects/def-xiaobird/rtm/image_output_dir'




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

        # #iDB test data
        if opt.test_dataset_iDB_10_500_80_120:
            sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
            sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_80_120.txt')
            sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

        elif opt.test_iDB_normalize_dataset_10_500_80_120:
            sub_list_MRI = os.path.join(sub_list_dir, 'normalize_iDB_MRI_80_120.txt')
            sub_list_CT = os.path.join(sub_list_dir, 'normalize_iDB_CT_80_120.txt')
            sub_list_seg = os.path.join(sub_list_dir, 'normalize_iDB_seg_80_120.txt')



        # train unpaired -10 500
        elif opt.test_dataset_train_10_500_80_120:
            sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
            sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_80_120.txt')#oasis_preprocess_80_120.txt or oasis_ncct_80_120.txt
            sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_80_120.txt')


        #
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

        imglist_MRI, imglist_CT,imglist_seg = sublist.equal_length_two_list(imglist_MRI, imglist_CT,imglist_seg);
        len_dataset = opt.len_dataset
        if len_dataset:
            imglist_MRI, imglist_CT, imglist_seg = imglist_MRI[:len_dataset], imglist_CT[:len_dataset], imglist_seg[:len_dataset]



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


        path_images = opt.path_images
        path_csv = opt.path_csv
        folder_name = opt.folder_name




        df = pd.DataFrame()

        for epoch_number in range(0,opt.max_epoch+2):
            opt.which_epoch = epoch_number
            opt.test_seg_output_dir = f'{path_images}/{folder_name}/{opt.name}/{opt.attribute}/epoch_{epoch_number}'
            cycle_output_dir = opt.test_seg_output_dir
            filename = f'{path_csv}/{folder_name}/{opt.name}/{opt.name}_{opt.attribute}_test_results.csv'
            mkdir(cycle_output_dir)
            mkdir(f'{path_csv}/{folder_name}/{opt.name}')

            print(filename,'**',cycle_output_dir)
            print('#testing images = %d' % dataset_size)
            model = create_model(opt)

            visualizer = Visualizer(opt)
            for i, data in enumerate(dataset):
                model.set_zero()
                model.set_input(data)
                model.test()
                img_path = model.get_image_paths()[0]
                # print('process image... %s' % img_path)
                if opt.MC_uncertainty :

                    # print('*** $$$$ @@@',epoch_number,' * ',i)
                    coef = model.get_coef()
                    # print(coef)
                    df_test = pd.DataFrame([coef])
                    df_test['epoch'] = epoch_number
                    df_test['data_number'] = i
                    # if opt.print_images_with_uncertainty and (i+1)%41 <35:
                    #     visuals = model.get_current_visuals()
                    #     image_paths_A, image_paths_B, image_paths_seg= model.get_image_paths()
                    #     visualizer.save_images_to_dir_uncertainty(cycle_output_dir, visuals, img_path)
                        # image_name = model.get_name()
                        # image = model.get_image()
                        # save_image_array(image, cycle_output_dir, image_name)
                    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
                        df_test.to_csv(filename, index=False)
                    else:
                        df_test.to_csv(filename, mode='a', header=False, index=False)


                elif not opt.MC_uncertainty:
                    # print('*&*&*&*%')
                    visuals = model.get_current_visuals()
                    image_paths_A, image_paths_B, image_paths_seg = model.get_image_paths()
                    visualizer.save_images_to_dir(cycle_output_dir, visuals, image_paths_A, image_paths_B, image_paths_seg)

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

        len_dataset = opt.len_dataset
        imglist_MRI, imglist_CT, imglist_seg = sublist.equal_length_two_list(imglist_MRI, imglist_CT, imglist_seg);
        if len_dataset:
            imglist_MRI, imglist_CT, imglist_seg = imglist_MRI[:len_dataset], imglist_CT[:len_dataset], imglist_seg[:len_dataset]

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






        df = pd.DataFrame()
        path_images = opt.path_images
        path_csv = opt.path_csv
        folder_name = opt.folder_name


        for epoch_number in range(0, opt.max_epoch + 1):  # max_epoch+1
            opt.which_epoch = epoch_number
            opt.test_seg_output_dir = f'{path_images}/{folder_name}/{opt.name}/{opt.attribute}/epoch_{epoch_number}'
            cycle_output_dir = opt.test_seg_output_dir
            filename = f'{path_csv}/{folder_name}/{opt.name}_{opt.attribute}_test_results.csv'
            print(f'{path_csv}/{folder_name}', '**', cycle_output_dir)

            mkdir(cycle_output_dir)
            mkdir(f'{path_csv}/{folder_name}')
            print('#testing images = %d' % dataset_size)
            model = create_model(opt)
            visualizer = Visualizer(opt)
            for i, data in enumerate(dataset):
                model.set_zero()
                model.set_input(data)
                model.test()
                # print('process image... %s' % img_path)
                if opt.MC_uncertainty:
                    # print('*** $$$$ @@@', epoch_number, ' * ', i)
                    coef = model.get_coef()
                    # print(coef)
                    df_test = pd.DataFrame([coef])
                    df_test['epoch'] = epoch_number
                    df_test['data_number'] = i

                    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
                        df_test.to_csv(filename, index=False)
                    else:
                        df_test.to_csv(filename, mode='a', header=False, index=False)


                elif not opt.MC_uncertainty:
                    # print('*&*&*&*%')
                    visuals = model.get_current_visuals()
                    image_paths_A, image_paths_B= model.get_image_paths()
                    visualizer.save_cycle_gan_images_to_dir(cycle_output_dir, visuals, image_paths_A, image_paths_B)
