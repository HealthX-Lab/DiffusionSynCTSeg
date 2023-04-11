import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from src.datasetscollection_model import DatasetsCollection
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.data import CacheDataset, DataLoader, decollate_batch
import random
from src.transformer_model import MyTransformer
from util.visualizer import Visualizer
from monai.metrics import DiceMetric,ConfusionMatrixMetric
import random
import numpy as np
from . import networks
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    Compose,
)
class TestChain:
    def __init__(self, opt, paths):
        # self.test_data = random.shuffle(paths)
        self.data_paths = paths
        self.option = opt
        self.confusion_matrix_list = {i: [] for i in self.option.ConfusionMatrixMetric}
        self.visualizer = Visualizer(self.option)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=True)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.confusion_matrix = ConfusionMatrixMetric(include_background=False,
                                                      metric_name=self.option.ConfusionMatrixMetric)
        self.metric_list_test = {'seg_B': {i: [] for i in self.option.ConfusionMatrixMetric}}
        self.dice_test = {'seg_B': []}



    def build_test_transformers(self):
        transformer_obj_test= MyTransformer(self.option,'Test')
        test_data_transformes = transformer_obj_test.get_data_transformer()
        test_ds = CacheDataset(data=self.data_paths,
                                transform=test_data_transformes,
                                cache_rate=self.option.cache_rate,
                                num_workers=self.option.dataLoader_num_workers)
        self.test_loader = DataLoader(test_ds, batch_size=self.option.batch_size)
        self.test_size = len(self.test_loader)

    def test_model(self):
        self.model.eval()
        self.counter = 0
        with torch.no_grad():
            for test_data in self.test_loader:
                self.model.set_input(test_data)
                self.model.calculate_inference()
                output_images = self.model.get_current_visuals()
                # seg_outputs = [self.post_trans(i) for i in decollate_batch(seg_pred)]
                self.counter = self.counter + 1
                self.calculate_metrics(output_images)
                self.visualize_images(output_images)

            self.aggrigate_metrics()
            self.reset_metrics()
        self.visualizer.log_test(self.metric_list_test)
        self.visualizer.log_test(self.dice_test)




    def build_chain(self):
        self.build_test_transformers()
        self.create_model()
        self.test_model()

    def create_model(self):
        self.model = None
        if self.option.operation == 'Test':
            from src.test_seg_model import TestSEGModel
            self.model = TestSEGModel()
        self.model.initialize(self.option)

    def calculate_metrics(self,output_images):
        self.confusion_matrix(y_pred=output_images['seg_B'], y=output_images['real_seg'])
        self.dice_metric(y_pred=output_images['seg_B'], y=output_images['real_seg'])

    def aggrigate_metrics(self):
        rec_seg_conf_matrix = self.confusion_matrix.aggregate()
        self.metric_list_test['seg_B'] = self.append_metric(self.metric_list_test['seg_B'],rec_seg_conf_matrix)

        seg_B_dice_loss = self.dice_metric.aggregate().item()
        self.dice_test['seg_B'].append(seg_B_dice_loss)

    def reset_metrics(self):
        self.dice_metric.reset()
        self.confusion_matrix.reset()

    def append_metric(self,metric_dict_image,conf_matrix):
        for i in range(len(self.option.ConfusionMatrixMetric)):
            metric_dict_image[self.option.ConfusionMatrixMetric[i]].append(
                conf_matrix[i].item())
        return metric_dict_image


def visualize_images(self, output_images):
    for i in range(len(output_images['seg_B'])):

        pred_seg_cpu = output_images['seg_B'][i].cpu().numpy()[0].transpose((1, 0, 2))
        real_seg_cpu = output_images['real_seg'][i].cpu().numpy()[0].transpose((1, 0, 2))

        self.visualizer.save_images(pred_seg_cpu, f'pred_seg_{self.counter}_batch_{i}', label=True, epoch=self.counter)
        self.visualizer.save_images(real_seg_cpu, f'real_seg_{self.counter}_batch_{i}', label=True, epoch=self.counter)

        if self.option.calculate_uncertainty:
            variance = output_images['variance'].cpu().numpy()[i].transpose((0, 2, 1, 3))
            self.visualizer.save_images(variance[0], f'heatmap_class_{0}_image_{self.counter}_batch_{i}', label=True,
                                        epoch=self.counter)
            self.visualizer.save_images(variance[1], f'heatmap_class_{1}_image_{self.counter}_batch_{i}', label=True,
                                        epoch=self.counter)
            self.visualizer.save_images(variance[2], f'heatmap_class_{2}_image_{self.counter}_batch_{i}', label=True,
                                        epoch=self.counter)
            self.visualizer.save_images(variance[3], f'heatmap_class_{3}_image_{self.counter}_batch_{i}', label=True,
                                        epoch=self.counter)
            self.visualizer.save_images(variance[4], f'heatmap_class_{4}_image_{self.counter}_batch_{i}', label=True,
                                        epoch=self.counter)

            entropy = output_images['entropy'].cpu().numpy()[i].transpose((1, 0, 2))
            self.visualizer.save_images(entropy, f'entropy_{self.counter}_batch_{i}', label=True, epoch=self.counter)


















