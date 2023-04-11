import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from src.datasetscollection_model import DatasetsCollection
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.data import CacheDataset, DataLoader
import random
from src.transformer_model import MyTransformer
from monai.networks import one_hot
from torchsummary import summary
from util.visualizer import Visualizer
from monai.metrics import DiceMetric,ConfusionMatrixMetric
import math
from numpy import inf
import nibabel as nib
import numpy as np

# from src2.model import Model
class TrainingChain:
    def __init__(self, opt, paths):
        self.data_dicts_train = []
        self.data_dicts_val = []
        self.data_paths = paths
        self.option = opt
        self.train_data, self.val_data = self.seprate_train_val()
        self.visualizer = Visualizer(self.option)
        self.best_loss = inf

        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.confusion_matrix = ConfusionMatrixMetric(include_background=False,
                                                 metric_name=self.option.ConfusionMatrixMetric)

        self.confusion_matrix_list = {i: [] for i in self.option.ConfusionMatrixMetric}
        image_types = [ 'fake_A','rec_A', 'idt_A',
                        'fake_B', 'rec_B', 'idt_B', 'seg_B']

        self.best_metric = -1
        self.best_metric_epoch = 0 # based on val data

        self.epoch_loss_train = {} # list of losses in end of each train epoch
        self.metric_list_train = {image_name:{i: [] for i in self.option.ConfusionMatrixMetric} for image_name in image_types}  # list of metrics in end of each train epoch


        self.metric_list_val = {'seg_B':{i: [] for i in self.option.ConfusionMatrixMetric}}
        self.dice_val = {'seg_B':[] }



    def build_transformers(self):
        transformer_obj_train = MyTransformer(self.option,'Train')
        train_data_transformes = transformer_obj_train.get_data_transformer()
        train_ds = CacheDataset(data=self.train_data,
                                transform=train_data_transformes,
                                cache_rate=self.option.cache_rate,
                                num_workers=self.option.dataLoader_num_workers)
        self.train_loader = DataLoader(train_ds, batch_size=self.option.batch_size)
        self.dataset_size = len(self.train_loader)
        transformer_obj_val = MyTransformer(self.option, 'val')
        val_data_transformes = transformer_obj_val.get_data_transformer()
        val_ds = CacheDataset(data=self.val_data,
                              transform=val_data_transformes,
                              cache_rate=self.option.cache_rate,
                              num_workers=self.option.dataLoader_num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=self.option.batch_size)


    def get_val_data(self):
        return self.val_data

    def get_train_data(self):
        return self.train_data

    def visualize_images(self,output_images,epoch,number=0):
        lambda_idt = self.option.identity
        image_types = ['real_A', 'fake_A', 'rec_A',
                       'real_B', 'fake_B', 'rec_B',
                       'seg_B', 'real_seg']
        if lambda_idt > 0:
            image_types.append('idt_A')
            image_types.append('idt_B')

        for i in range(len(output_images['seg_B'])):
            for type in image_types:
                if 'seg' not in type:
                    print('##########output_images[type]##########', type,np.shape(output_images[type][i].cpu().numpy()[0]))
                    cpu_image = output_images[type][i].cpu().numpy()[0].transpose((1, 0, 2))
                    self.visualizer.save_images(cpu_image, f'{type}_val_{number}_epoch_{epoch}_batch_{i}',epoch=epoch)
                else:
                    print('##########output_images[type] seg##########', type,np.shape(output_images[type][i].cpu().numpy()[0]))
                    cpu_image = output_images[type][i].cpu().numpy()[0].transpose((1, 0, 2))
                    self.visualizer.save_images(cpu_image, f'{type}_val_{number}_epoch_{epoch}_batch_{i}',label=True,epoch=epoch)



        image_types.remove('real_A')
        image_types.remove('real_B')
        image_types.remove('real_seg')

        if self.option.calculate_uncertainty:
            for i in range(len(output_images['seg_B'])):
                for type in image_types:
                    print('########## output_images[variance][type]##########', np.shape( output_images['variance'][type][i]))
                    cpu_variance_image = output_images['variance'][type][i].cpu().numpy().transpose((0,2,1,3))
                    if 'seg' in type:
                        print('entropy .cpu().numpy()', np.shape(output_images['entropy'][type].cpu().numpy()))

                        entropy = output_images['entropy'][type].cpu().numpy()[i].transpose((1, 0, 2))
                        print('entropy', np.shape(entropy))
                        self.visualizer.save_images(entropy, f'entropy_{number}_epoch_{epoch}_batch_{i}',label=True,epoch=epoch)


                        print('seg',np.shape(cpu_variance_image[i]))
                        for j in range(self.option.num_classes):
                            self.visualizer.save_images(cpu_variance_image[j], f'heatmap_{type}_val_{number}_class_{j}_epoch_{epoch}_batch_{i}',label=True,epoch=epoch)
                    else:
                        print(' not seg', np.shape(cpu_variance_image[0]))
                        self.visualizer.save_images(cpu_variance_image[0], f'heatmap_{type}_val_{number}_epoch_{epoch}_batch_{i}',epoch=epoch)





    def eval_model(self,epoch):
        self.model.eval()
        with torch.no_grad():
            for val_data in self.val_loader:
                self.model.set_input(val_data)
                self.model.evaluation_step()
                errors = self.model.get_current_errors(is_train=False)
                output_images = self.model.get_current_visuals()

                self.calculate_metrics(output_images)


            self.aggrigate_metrics()
            self.reset_metrics()

        # reset the status for next validation round
        if self.dice_val['seg_B'][-1] < self.best_loss:
            self.best_loss = self.dice_val['seg_B'][-1]
            self.visualizer.log_loss(
                f"epoch {epoch}/{self.option.max_epochs}\n"
                f" dice loss {self.best_loss} \n"
                f"model errors {errors}")
            self.visualizer.log_model(
                f"saving the best model (epoch {epoch}"
                f"with dice loss {self.best_loss}\n ")
            self.model.save(f'best_dice_loss_{epoch}')


    def train_model(self):
        total_steps = 0
        best_loss = math.inf
        epoch_start_time = time.time()
        epoch_start = 0 if self.option.continue_train == False \
            else self.option.which_epoch

        for epoch in range(epoch_start+1,self.option.max_epochs):
            self.visualizer.log_model("-" * 10)
            self.model.train()
            self.visualizer.log_model(f"epoch {epoch}/{self.option.max_epochs}\n")

            epoch_iter = 0


            for data in self.train_loader:

                iter_start_time = time.time()
                start = time.strftime
                total_steps += self.option.batch_size
                epoch_iter += self.option.batch_size

                self.visualizer.log_model(f"epoch iteration {epoch_iter}/{len(self.train_loader)}  "
                                          f"in epoch {epoch}   "
                                          f"start time: {start}\n")

                self.model.set_input(data)
                self.model.optimize_parameters()

                errors = self.model.get_current_errors()
                mean_errors = sum(errors.values()) / len(errors)

                t = (time.time() - iter_start_time) / self.option.batch_size

                if total_steps % self.option.print_freq == 0:
                    self.visualizer.log_loss(
                        f"epoch {epoch}/{self.option.max_epochs}\n"
                        f" epoch iteration {epoch_iter}/{len(self.train_loader)} \n taken time for each epoch: {t}\n"
                        f" errors {errors}\n")
                    self.visualizer.log_model(
                        f"saving the latest model (epoch {epoch}, total_steps {total_steps})\n "
                        f"with loss {errors}\n mean error {mean_errors}\n taken time for each epoch: {t}\n")
                    self.model.save(f'iteration_{epoch_iter}_epoch_{epoch}')

                if mean_errors < best_loss:
                    best_loss = mean_errors
                    self.visualizer.log_loss(
                        f"epoch {epoch}/{self.option.max_epochs}\n"
                        f" epoch iteration {epoch_iter}/{len(self.train_loader)} \n taken time for each epoch: {t}\n"
                        f" errors {errors}\n")
                    self.visualizer.log_model(
                        f"saving the best model (epoch {epoch}, total_steps {total_steps})\n "
                        f"with loss {errors}\n mean error {mean_errors}")
                    self.model.save(f'best_mean_error_{epoch}')


            self.visualizer.log_model(
                f"saving the model at end of epoch (epoch {epoch}, total_steps {total_steps})\n "
                f"with loss {errors}\n mean error {mean_errors}")
            self.model.save(f'epoch_{epoch}')

            self.eval2_model(epoch)
            self.epoch_loss_train = self.append_epoch_loss(self.model.get_current_errors(), self.epoch_loss_train)

        self.visualizer.log_model(f'End of epoch {epoch} Time Taken {time.time() - epoch_start_time}')
        self.visualize_metrics()

    def visualize_metrics(self):

        for model, value in self.epoch_loss_train.items():
            self.visualizer.plot_metrics(value,f"loss_train_for_{model}_model")

        # for image_name, value in self.metric_list_train.items() :
        #     for metric, metric_value in value.items():
        #         self.visualizer.plot_metrics(metric_value,f"metric_{metric}_for_{image_name}_images_in_train")

        for image_name, value in self.metric_list_val.items() :
            for metric, metric_value in value.items():
                self.visualizer.plot_metrics(metric_value,f"metric_{metric}_for_{image_name}_images_in_val")

        for image_name, value in self.dice_val.items() :
            self.visualizer.plot_metrics(value,f"dice_loss_for_{image_name}_images_in_val")

    def build_chain(self):
        self.build_transformers()
        self.create_model()
        self.train_model()

    def seprate_train_val(self):
        random.shuffle(self.data_paths)
        number_val = int(len(self.data_paths) * self.option.val_rate)
        val_data = self.data_paths[:number_val]
        train_data = self.data_paths[number_val:]
        return train_data, val_data

    def create_model(self):
        self.model = None
        if self.option.model_type == 'GAN':
            from .cycle_gan_model import CycleSEGModel
            self.model = CycleSEGModel()

        self.model.initialize(self.option)
        self.epoch_loss_train = {i: [] for i in self.model.get_loss_name()}
        self.model.print_network(self.visualizer)
        self.visualizer.log_model(f"model {self.option.model_type} was created\n")

    def append_epoch_loss(self,loss_dic,target_dic):
        for key, value in loss_dic.items():
            target_dic[key].append(value)
        return target_dic

    def append_metric(self,metric_dict_image,conf_matrix):
        for i in range(len(self.option.ConfusionMatrixMetric)):
            metric_dict_image[self.option.ConfusionMatrixMetric[i]].append(
                conf_matrix[i].item())
        return metric_dict_image

    def calculate_metrics(self,output_images):
        self.confusion_matrix(y_pred=output_images['seg_B'], y=output_images['real_seg'])
        self.dice_metric(y_pred=output_images['seg_B'], y=output_images['real_seg'])

    def aggrigate_metrics(self):
        rec_seg_conf_matrix = self.confusion_matrix.aggregate()
        self.metric_list_val['seg_B'] = self.append_metric(self.metric_list_val['seg_B'],rec_seg_conf_matrix)

        seg_B_dice_loss = self.dice_metric.aggregate().item()
        self.dice_val['seg_B'].append(seg_B_dice_loss)

    def reset_metrics(self):
        self.dice_metric.reset()
        self.confusion_matrix.reset()

    def eval2_model(self,epoch):
        self.model.eval()
        number = 0
        with torch.no_grad():
            for val_data in self.val_loader:
                number = number + 1
                print('I am here *********',number)
                self.model.set_input(val_data)
                self.model.calculate_evaluation()

                errors = self.model.get_current_errors(is_train=False)
                output_images = self.model.get_current_visuals()
                print('val_errors',errors)

                self.calculate_metrics(output_images)
                self.visualize_images(output_images,epoch,number)


            self.aggrigate_metrics()
            self.reset_metrics()

        # reset the status for next validation round
        if self.dice_val['seg_B'][-1] < self.best_loss:
            self.best_loss = self.dice_val['seg_B'][-1]
            self.visualizer.log_loss(
                f"epoch {epoch}/{self.option.max_epochs}\n"
                f" dice loss {self.best_loss} \n"
                f"model errors {errors}")
            self.visualizer.log_model(
                f"saving the best model (epoch {epoch}"
                f"with dice loss {self.best_loss}\n ")
            self.model.save(f'beast_dice_loss_{epoch}')

















