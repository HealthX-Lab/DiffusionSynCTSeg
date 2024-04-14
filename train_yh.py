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


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


opt = TrainOptions().parse()

# Method = 'ImageOnly'
Method = opt.yh_data_model

# # paired models
# raw_MRI_dir = '/home/rtm/scratch/2D/MRI'
# raw_MRI_seg_dir = '/home/rtm/scratch/2D/seg'
# raw_CT_dir = '/home/rtm/scratch/2D/clip_min_max_preprocess/iDB'#'/home/rtm/scratch/2D/clip_min10_max200_std_preprocess/iDB'#'/home/rtm/scratch/2D/clip_min_max_preprocess/iDB'
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

    cycle_output_dir = opt.test_seg_output_dir
    # if not os.path.exists(cycle_output_dir):
    #     cycle_output_dir = '/scratch/huoy1/projects/DeepLearning/Cycle_Deep/Output/CycleTest'


    # mkdir(sub_list_dir)

    # sub_list_MRI = os.path.join(sub_list_dir, 'test_MRI_iDB_paired.txt')
    # sub_list_CT = os.path.join(sub_list_dir, 'test_CT_iDB_paired.txt')#test_CT_iDB_paired_m10_200.txt   test_CT_iDB_paired.txt
    # sub_list_seg = os.path.join(sub_list_dir, 'test_seg_iDB_paired.txt')

    sub_list_MRI = os.path.join(sub_list_dir, 'test_MRI_list.txt')
    sub_list_CT = os.path.join(sub_list_dir, 'test_CT_list.txt')
    sub_list_seg = os.path.join(sub_list_dir, 'test_seg_sublist.txt')

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


    print('#testing images = %d' % dataset_size)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        label = 'real_A'
        image_numpy = visuals[label]
        print(f"Minimum value in visualisation A in train_yh: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
        print(f"Maximum value in visualisation A train_yh: {np.max(image_numpy)}")

        label = 'real_B'
        image_numpy = visuals[label]
        print(f"Minimum value in visualisation B in train_yh: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
        print(f"Maximum value in visualisation B train_yh: {np.max(image_numpy)}")


        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images_to_dir(cycle_output_dir, visuals, img_path)

elif TrainOrTest == 'TestSeg':
    print('in test seg***************')
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.isTrain = False
    opt.phase = 'test'
    opt.no_dropout = True
    seg_output_dir = opt.test_seg_output_dir

    opt.test_CT_dir = opt.test_CT_dir

    if opt.custom_sub_dir == 1:
        sub_list_dir = os.path.join(seg_output_dir,'sublist')
        mkdir(sub_list_dir)
    print('path test dataset ',sub_list_dir,'test_CT_list.txt')
    test_img_list_file = os.path.join(sub_list_dir,'test_CT_list.txt')
    opt.imglist_testCT = sublist.dir2list(opt.test_CT_dir, test_img_list_file)
    opt.imglist_testCT = sublist.dir2list(opt.test_CT_dir, test_img_list_file)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        print('i am here in test seg ready to print images')
        visualizer.save_seg_images_to_dir(seg_output_dir, visuals, img_path)


elif TrainOrTest == 'Train':
    print('in the train')

    # mkdir(sub_list_dir)
    # paired
    # sub_list_MRI = os.path.join(sub_list_dir, 'iDB_MRI_preprocess_80_120.txt')
    # sub_list_CT = os.path.join(sub_list_dir, 'iDB_preprocess_80_120.txt')
    # sub_list_seg = os.path.join(sub_list_dir, 'iDB_seg_preprocess_80_120.txt')

    #unpaired -10 500
    sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri_expend_80_120.txt')
    sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_80_120.txt')
    sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_80_120.txt')


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
    opt.raw_MRI_dir = raw_MRI_dir
    opt.raw_MRI_seg_dir = raw_MRI_seg_dir
    opt.raw_CT_dir = raw_CT_dir
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
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            # print('*****before optimization*****')
            model.optimize_parameters(epoch)
            # print('*****after optimization*****')

            # if total_steps % opt.display_freq == 0:
            #     visualizer.display_current_results(model.get_current_visuals(), epoch)
            # save model if this is the best result
            # model.earlyStopping(epoch, patience=opt.patience)
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                print(errors)
                new_df = pd.DataFrame([errors])
                # df = df.append(new_df, ignore_index=True)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(f'/home/rtm/scratch/model_outputs/csvfiles/2D/{opt.name}.csv', index=False)
                t = (time.time() - iter_start_time) / opt.batchSize
                print('errors:**',errors)
                # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                # if opt.display_id > 0:
                #     visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()


