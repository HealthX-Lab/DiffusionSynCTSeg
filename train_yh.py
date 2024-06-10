# This script sets up and runs the training process for a deep learning model
# using specified options, data directories, and training configurations.
# Note: The sublist directory must contain text files listing path to all relevant
# training, validation, and testing data.

import time
import os
import sublist
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torch
import pandas as pd
from util.util import append_data_to_csv

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


opt = TrainOptions().parse()

# Method = 'ImageOnly'
Method = opt.yh_data_model



# unpaired models
raw_MRI_dir = 'path-to-2D-MRI'
raw_MRI_seg_dir = 'path-to-2D-label'
raw_CT_dir = 'path-to-2D-CT'
sub_list_dir = 'path-to-2D-sublists'# txt files that contain name of all 2D images with their paths




ngpus_per_node = torch.cuda.device_count()
local_rank = int(os.environ.get("SLURM_LOCALID"))
rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
print('******* rank  ', rank,local_rank,flush= True)
opt.gpu_ids = [local_rank]
print('***** gpu ids', opt.gpu_ids , flush=True)



TrainOrTest = opt.yh_run_model #'Train' #
df = pd.DataFrame(columns = ['D_A','G_A','Cyc_A','D_B', 'G_B','Cyc_B','Seg'])
val_df = pd.DataFrame(columns = ['D_A','G_A','Cyc_A','D_B', 'G_B','Cyc_B','Seg'])





if TrainOrTest == 'Train':
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
        sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_80_120.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_80_120.txt')



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
        sub_list_CT = os.path.join(sub_list_dir, 'oasis_ncct_85_100.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'sublist_seg_expend_85_100.txt')

    elif opt.train_dataset_IXI_neuro_all_slice:
        sub_list_MRI = os.path.join(sub_list_dir, 'all_slices_MRI_IXI_preprocess.txt')
        sub_list_CT = os.path.join(sub_list_dir, 'sublist_all_CT_train.txt')
        sub_list_seg = os.path.join(sub_list_dir, 'all_slices_MRI_label_IXI_preprocess.txt')


    opt.raw_MRI_dir = raw_MRI_dir
    opt.raw_MRI_seg_dir = raw_MRI_seg_dir
    opt.raw_CT_dir = raw_CT_dir

    if opt.model=='cut' or opt.model=='sb':
        optimize_time = 0.1
        opt.eval = False
        opt.eval_step = False

    # If evaluation step is enabled, set validation lists
    if opt.eval_step:
        sub_list_MRI_val = os.path.join(sub_list_dir, 'val_normalize_iDB_MRI_80_120.txt')
        sub_list_CT_val = os.path.join(sub_list_dir, 'val_normalize_iDB_CT_80_120.txt')
        sub_list_seg_val = os.path.join(sub_list_dir, 'val_normalize_iDB_seg_80_120.txt')

        imglist_MRI_val = sublist.dir2list(raw_MRI_dir, sub_list_MRI_val)
        imglist_CT_val = sublist.dir2list(raw_CT_dir, sub_list_CT_val)
        imglist_seg_val = sublist.dir2list(raw_MRI_seg_dir, sub_list_seg_val)


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
        val_dataset_size = len(val_data_obj)




    imglist_MRI = sublist.dir2list(raw_MRI_dir, sub_list_MRI)
    imglist_CT = sublist.dir2list(raw_CT_dir, sub_list_CT)
    imglist_seg = sublist.dir2list(raw_MRI_seg_dir, sub_list_seg)

    # Ensure equal length of image lists
    imglist_MRI, imglist_CT,imglist_seg = sublist.equal_length_two_list(imglist_MRI, imglist_CT,imglist_seg);
    print('after sublist')

    # Assign image lists to options
    opt.imglist_MRI = imglist_MRI
    opt.imglist_CT = imglist_CT
    opt.imglist_seg = imglist_seg

    # Set weights for the cross-entropy loss to balance the segmentation of ventricles and background pixels,
    # based on the relative abundance of ventricle pixels
    opt.crossentropy_weight = [1,30]

    # Load dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    if opt.model=='cut' or opt.model=='sb':
        data_loader2 = CreateDataLoader(opt)
        dataset2 = data_loader2.load_data()
        dataset_size2 = len(data_loader2)



    # Create model and visualizer
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    print('#model created')
    epoch_start_time = time.time()
    epoch_iter = 0
    total_iters = 0
    if not (opt.model == 'cut' or opt.model == 'sb'):
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            print('lr: ',opt.lr)

            # Adjust the weight for segmentation relative to translation during training
            if opt.segmentation_linear_increase:
                opt.weight_segmentation_in_GAN = 0.00001 * epoch / 25
            elif opt.segmentation_threshold_increase:
                if epoch<5:
                    opt.weight_segmentation_in_GAN = 0
                else:
                    opt.weight_segmentation_in_GAN = 0.00001 * (epoch - 4)/21
            print('opt.weight_segmentation_in_GAN : ',opt.weight_segmentation_in_GAN)

            # Iterate through the dataset
            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize

                # Set the input data for the model
                model.set_input(data, i)
                # Optimize the model parameters
                model.optimize_parameters(i, epoch)

                if total_steps % opt.print_freq == 0:
                    print('total_steps: ',total_steps)

                    # Retrieve current training errors and append them to the CSV file
                    errors = model.get_current_errors()
                    csv_file_path = f'{opt.test_seg_output_dir}.csv'
                    append_data_to_csv(csv_file_path, errors)

                    # Update the DataFrame with new errors and save it to a CSV file
                    new_df = pd.DataFrame([errors])
                    df = pd.concat([df, new_df], ignore_index=True)
                    df.to_csv(f'./results/{opt.name}.csv', index=False)

                    t = (time.time() - iter_start_time) / opt.batchSize
                    # Perform evaluation step if enabled and at the specified interval
                    if opt.eval_step and total_steps % (opt.print_freq) == 0:
                        for val_num, val_data in enumerate(val_dataset):
                            model.test(val_data)
                            val_result = model.get_val_result()
                            model.save_best_model()
                            val_csv_file_path = f'{opt.test_seg_output_dir}_val.csv'
                            append_data_to_csv(val_csv_file_path, val_result)
                            new_val_df = pd.DataFrame([val_result])
                            val_df = pd.concat([val_df, new_val_df], ignore_index=True)
                            val_df.to_csv(f'./save_best/{opt.name}_validation.csv',
                                      index=False)

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


    if  opt.model == 'cut' or opt.model == 'sb':
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0
            print('lr: ', opt.lr)

            for i, (data, data2) in enumerate(zip(dataset, dataset2)):
                iter_start_time = time.time()
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_iters += opt.batchSize
                epoch_iter += opt.batchSize
                optimize_start_time = time.time()
                batch_size = data["A"].size(0)

                if epoch == opt.epoch_count and i == 0:
                    model.data_dependent_initialize(data, data2)
                    model.setup(opt)  # regular setup: load and print networks; create schedulers
                    model.parallelize()
                model.set_input(data, data2, i)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()

                optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time


                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    print(losses)
                    csv_file_path = f'{opt.test_seg_output_dir}.csv'
                    append_data_to_csv(csv_file_path, losses)


                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    print(opt.name)  # it's useful to occasionally show the experiment name on console
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (
                epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            model.update_learning_rate()  # update learning rates at the end of every epoch.


