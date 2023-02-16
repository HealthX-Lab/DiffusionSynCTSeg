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
        image_types = [ 'fake_B', 'rec_A', 'idt_B','seg_B',
                     'fake_A', 'rec_B', 'idt_A', 'real_seg']

        self.best_metric = -1
        self.best_metric_epoch = 0 # based on val data

        self.epoch_loss_train = {} # list of losses in end of each train epoch
        self.metric_list_train = {image_name:{i: [] for i in self.option.ConfusionMatrixMetric} for image_name in image_types}  # list of metrics in end of each train epoch


        self.metric_list_val = {image_name:{i: [] for i in self.option.ConfusionMatrixMetric} for image_name in image_types}
        self.dice_metric_val = {image_name:[] for image_name in image_types}







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

    def eval_model(self,epoch):
        self.model.eval()
        with torch.no_grad():
            for val_data in self.val_loader:
                self.model.set_input(val_data)
                self.model.evaluation_step()
                outputs = self.model.get_current_visuals()

                self.metric_list_val = calculate_conf_matrix_based_on_image(self,self.metric_list_val, outputs)
                self.dice_metric_val = calculate_dice_metric_based_on_image(self, self.dice_metric_val, outputs)

                self.dice_metric.reset()
                self.confusion_matrix.reset()

        sum_conf_matrix_val = self.sum_metrics(self.metric_list_val)
        sum_dice_loss_val = self.sum_loss(self.dice_metric_val)
        sum_over_images = sum(sum_dice_loss_val.values())


        # reset the status for next validation round
        if sum_over_images < self.best_loss:
            self.best_loss = sum_over_images
            self.visualizer.log_loss(
                f"epoch {epoch + 1}/{self.option.max_epochs}\n"
                f" dice loss {sum_dice_loss_val} and sum {sum_over_images}\n")
            self.visualizer.log_model(
                f"saving the best model (epoch {epoch + 1}"
                f"with dice loss {sum_dice_loss_val}\n confusion matrix {sum_conf_matrix_val}")
            self.model.save(f'beast_dice_loss_{epoch}')


    def train_model(self):
        total_steps = 0
        best_loss = math.inf
        epoch_start_time = time.time()

        for epoch in range(self.option.max_epochs):
            self.visualizer.log_model("-" * 10)
            self.model.train()
            self.visualizer.log_model(f"epoch {epoch + 1}/{self.option.max_epochs}\n")

            epoch_iter = 0
            errors = 0
            mean_errors = 0

            for data in self.train_loader:
                iter_start_time = time.time()
                start = time.strftime
                total_steps += self.option.batch_size
                epoch_iter += self.option.batch_size
                print('before set input ', flush=True)

                self.visualizer.log_model(f"epoch iteration {epoch_iter}/{len(self.train_loader)}  "
                                          f"in epoch {epoch+1}   "
                                          f"start time: {start}\n")

                self.model.set_input(data)
                print('model input set finished', flush=True)
                self.model.optimize_parameters()

                errors = self.model.get_current_errors()
                mean_errors = sum(errors.values()) / len(errors)

                t = (time.time() - iter_start_time) / self.option.batch_size

                if total_steps % self.option.print_freq == 0:
                    self.visualizer.log_loss(
                        f"epoch {epoch + 1}/{self.option.max_epochs}\n"
                        f" epoch iteration {epoch_iter}/{len(self.train_loader)} \n taken time for each epoch: {t}\n"
                        f" errors {errors}\n")
                    self.visualizer.log_model(
                        f"saving the latest model (epoch {epoch + 1}, total_steps {total_steps})\n "
                        f"with loss {errors}\n mean error {mean_errors}\n taken time for each epoch: {t}\n")
                    self.model.save(f'iteration_{epoch_iter}_epoch_{epoch+1}')

                if mean_errors < best_loss:
                    best_loss = mean_errors
                    self.visualizer.log_loss(
                        f"epoch {epoch + 1}/{self.option.max_epochs}\n"
                        f" epoch iteration {epoch_iter}/{len(self.train_loader)} \n taken time for each epoch: {t}\n"
                        f" errors {errors}\n")
                    self.visualizer.log_model(
                        f"saving the best model (epoch {epoch+1}, total_steps {total_steps})\n "
                        f"with loss {errors}\n mean error {mean_errors}")
                    self.model.save(f'beast_mean_error_{epoch}')


            self.visualizer.log_model(
                f"saving the model at end of epoch (epoch {epoch}, total_steps {total_steps})\n "
                f"with loss {errors}\n mean error {mean_errors}")
            self.model.save(f'epoch_{epoch}')

            self.eval_model(epoch)
            self.epoch_loss_train = self.append_epoch_loss(self.model.get_current_errors(),self.epoch_loss_train)

        self.visualizer.log_model(f'End of epoch {epoch} Time Taken {time.time() - epoch_start_time}')
        self.visualize_metrics()

    def visualize_metrics(self):

        for key, value in self.epoch_loss_train:
            self.visualizer.plot_metrics(value,f"loss train for {key} model")
        for key, value in self.metric_list_train :
            for metric, metric_value in value:
                self.visualizer.plot_metrics(metric_value,f"metric {metric} for {key} images in train")
        for key, value in self.metric_list_val :
            for metric, metric_value in value:
                self.visualizer.plot_metrics(metric_value,f"metric {metric} for {key} images in val")
        for key, value in self.metric_list_val :
            self.visualizer.plot_metrics(value,f"dice loss for {key} images in val")

    def build_chain(self):
        self.build_transformers()
        self.create_model()
        self.train_model()

        self.visualizer.plot_metrics(self.epoch_loss)
        self.visualizer.plot_metrics(self.confusion_matrix_list)
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
        self.epoch_loss_train = {i: [] for i in self.model.get_networks_name()}
        self.model.print_network(self.visualizer)
        self.visualizer.log_model(f"model {self.option.model_type} was created\n")
    def append_epoch_loss(self,loss_dic,target_dic):
        for key, value in loss_dic:
            target_dic[key].append(value)
        return target_dic
    def append_metric(self,metric_dict,conf_matrix):
        for i in len(self.option.ConfusionMatrixMetric):
            metric_dict[self.option.ConfusionMatrixMetric[i]].append(
                conf_matrix.aggregate()[i].item())
        return conf_matrix


    def calculate_conf_matrix_based_on_image(self,metric_dict,output_images):
        fake_A_matrix = self.confusion_matrix(y_pred=output_images['fake_A'], y=output_images['real_A'])
        metric_dict['fake_A'] = append_metric(metric_dict['fake_A'],fake_A_matrix)

        fake_B_matrix = self.confusion_matrix(y_pred=output_images['fake_B'], y=output_images['real_B'])
        metric_dict['fake_B'] = append_metric(metric_dict['fake_B'], fake_B_matrix)

        rec_A_matrix = self.confusion_matrix(y_pred=output_images['rec_A'], y=output_images['real_A'])
        metric_dict['rec_A'] = append_metric(metric_dict['rec_A'], rec_A_matrix)

        rec_B_matrix = self.confusion_matrix(y_pred=output_images['rec_B'], y=output_images['real_B'])
        metric_dict['rec_B'] = append_metric(metric_dict['rec_B'], rec_B_matrix)

        seg_B_matrix = self.confusion_matrix(y_pred=output_images['seg_B'], y=output_images['real_seg'])
        metric_dict['seg_B'] = append_metric(metric_dict['seg_B'], seg_B_matrix)
        return metric_dict

    def calculate_dice_metric_based_on_image(self,dice_metric_dict,output_images):
        fake_A_dice_loss = self.dice_metric(y_pred=output_images['fake_A'], y=output_images['real_A'])
        dice_metric_dict['fake_A'].append(fake_A_dice_loss)

        fake_B_dice_loss = self.dice_metric(y_pred=output_images['fake_B'], y=output_images['real_B'])
        dice_metric_dict['fake_B'].append(fake_B_dice_loss)

        rec_A_dice_loss = self.dice_metric(y_pred=output_images['rec_A'], y=output_images['real_A'])
        dice_metric_dict['rec_A'].append(rec_A_dice_loss)

        rec_B_dice_loss = self.dice_metric(y_pred=output_images['rec_B'], y=output_images['real_B'])
        dice_metric_dict['rec_B'].append(rec_B_dice_loss)

        seg_B_dice_loss = self.dice_metric(y_pred=output_images['seg_B'], y=output_images['real_seg'])
        dice_metric_dict['seg_B'].append(seg_B_dice_loss)
        return metric_dict

    def sum_metrics(self, metrics_val):
        result = dict.fromkeys(metrics_val, {})
        for key, value in metrics_val:
            metric = value.keys()
            result[key][metric] = sum(value.values())
        return result

    def sum_loss(self, metrics_val):
        result = dict.fromkeys(metrics_val, None)
        for key, value in metrics_val:
            result[key] = sum(value)
        return result















