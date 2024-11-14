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
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"An error occurred while creating directories: {e}")


opt = TrainOptions().parse()

# Method = 'ImageOnly'
Method = opt.yh_data_model


params = [
    {
        'name': 'UNSB_Seg',
        'which_model_netG': 'resnet_9blocks_cond',
        'which_model_netSeg': 'R2AttU_Net',
        'model': 'sb',
        'GaussianBlur': 'False',
        'Local_Histogram_Equalization': 'False',
        'Histogram_Equalization': 'False',
        'B_normalization': 'True',
        'min_max_normalize': 'False',
        'yh_run_model': 'Test',
        'MC_uncertainty': 'False',
        'num_samples_uncertainty': 2,
        'print_images_with_uncertainty': 'True',
        'len_dataset': 5,
        'attribute': 'sb',
        'path_csv': '/home/rtm/scratch/model_outputs/csvfiles',
        'path_images': '/home/rtm/scratch/model_outputs/images',
        'model_seg': '2d',
        'checkpoints_dir': './checkpoints',

    },

]

for param_dict in params:

    for key, val in param_dict.items():
        if val in ['True', 'False']:  # if the value is a string 'True'/'False'
            val = bool(strtobool(val))
        setattr(opt, key, val)

    args = vars(opt)
    opt.eval = True

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # unpaired models
    raw_MRI_dir = 'path-to-2D-MRI test dataset'
    raw_MRI_seg_dir = 'path-to-2D-label test dataset'
    raw_CT_dir = 'path-to-2D-CT test dataset'
    sub_list_dir = 'path-to-2D-sublists test dataset'



    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
    print('***** rank  ', rank, local_rank, flush=True)
    opt.gpu_ids = [local_rank]
    print('***** gpu ids', opt.gpu_ids, flush=True)


    TrainOrTest = opt.yh_run_model


    # evaluation
    if TrainOrTest == 'Test':
        print('in test***************')
        opt.which_epoch =  -1

        opt.nThreads = 1
        opt.batchSize = 1
        opt.serial_batches = True  #disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True
        opt.isTrain = False
        opt.phase = 'test'


        sub_list_MRI = os.path.join(sub_list_dir, 'iDB_mri_preprocess_80_120_complete.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'iDB_CT_preprocess_80_120_complete.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'iDB_label_preprocess_80_120_complete.txt')

        imglist_MRI = sublist.dir2list(raw_MRI_dir, sub_list_MRI)
        imglist_CT = sublist.dir2list(raw_CT_dir, sub_list_CT)
        imglist_seg = sublist.dir2list(raw_MRI_seg_dir, sub_list_seg)

        imglist_MRI, imglist_CT, imglist_seg = sublist.equal_length_two_list(imglist_MRI, imglist_CT, imglist_seg);
        len_dataset = opt.len_dataset
        if len_dataset:
            imglist_MRI, imglist_CT, imglist_seg = imglist_MRI[:len_dataset], imglist_CT[:len_dataset], imglist_seg[
                                                                                                        :len_dataset]



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
        print('#testing images = %d' % dataset_size)

        path_images = opt.path_images
        path_csv = opt.path_csv

        df = pd.DataFrame()
        filename = f'{path_csv}/{opt.name}/{opt.name}_{opt.attribute}_test_results.csv'
        mkdir(f'{path_csv}/{opt.name}')
        print('csv path: ', filename)

        opt.test_seg_output_dir = f'{path_images}/{opt.name}/{opt.attribute}'
        mkdir(opt.test_seg_output_dir)
        print('image path: ', opt.test_seg_output_dir)





        model = create_model(opt)
        visualizer = Visualizer(opt)

        for i, data in enumerate(dataset):
            if i == 0:
                model.data_dependent_initialize(data, data)
                model.setup(opt)
                model.parallelize()

            model.set_zero()
            model.set_input(data, None, i)
            model.test()

            if opt.MC_uncertainty:

                if opt.model_seg == '2d':
                    coef = model.get_coef()
                    df_test = pd.DataFrame([coef])
                    df_test['data_number'] = i
                    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
                        df_test.to_csv(filename, index=False)
                    else:
                        df_test.to_csv(filename, mode='a', header=False, index=False)
                if opt.model_seg == '3d' and (i + 1) % 41 == 0:
                    coef = model.get_3dcoef()
                    df_test = pd.DataFrame(coef)
                    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
                        df_test.to_csv(filename, index=False)
                    else:
                        df_test.to_csv(filename, mode='a', header=False, index=False)
                if opt.print_images_with_uncertainty:
                    visuals = model.get_current_visuals()
                    image_paths_A = model.get_image_paths()
                    visualizer.save_images_to_dir_uncertainty(opt.test_seg_output_dir, visuals, image_paths_A)




            elif not opt.MC_uncertainty:
                visuals = model.get_current_visuals()
                visualizer.save_images(visuals, opt.test_seg_output_dir, i)


