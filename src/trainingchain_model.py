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
# from util.visualizer import Visualizer
# from src2.model import Model
class TrainingChain:
    def __init__(self, opt, paths):
        self.data_dicts_train = []
        self.data_dicts_val = []
        self.data_paths = paths
        self.option = opt
        self.train_data, self.val_data = self.seprate_train_val()

    def build_transformers(self):
        transformer_obj_train = MyTransformer(self.option,'Train')
        train_data_transformes = transformer_obj_train.get_data_transformer()
        train_ds = CacheDataset(data=self.train_data,
                                transform=train_data_transformes,
                                cache_rate=self.option.cache_rate,
                                num_workers=self.option.dataLoader_num_workers)
        self.train_loader = DataLoader(train_ds, batch_size=self.option.batch_size)
        dataset_size = len(self.train_loader)
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


    def build_chain(self):
        self.build_transformers()
        model = self.create_model()
        # total_steps = 0
        #
        # for epoch in range(self.opt.epoch_count, self.opt.niter + self.opt.niter_decay + 1):
        #     epoch_start_time = time.time()
        #     epoch_iter = 0
        #     for data in self.train_loader:
        #         iter_start_time = time.time()
        #         total_steps += self.opt.batchSize
        #         epoch_iter += self.opt.batchSize
        #         model.set_input(data)
        #         model.optimize_parameters()
        #         visualizer = Visualizer(self.opt)
        #
        #         if total_steps % self.opt.display_freq == 0:
        #             visualizer.display_current_results(model.get_current_visuals(), epoch)
        #
        #         if total_steps % self.opt.print_freq == 0:
        #             errors = model.get_current_errors()
        #             t = (time.time() - iter_start_time) / self.opt.batchSize
        #             visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        #             if self.opt.display_id > 0:
        #                 visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, self.opt, errors)
        #
        #         if total_steps % self.opt.save_latest_freq == 0:
        #             print('saving the latest model (epoch %d, total_steps %d)' %
        #                   (epoch, total_steps))
        #             model.save('latest')
        #
        #     if epoch % self.opt.save_epoch_freq == 0:
        #         print('saving the model at the end of epoch %d, iters %d' %
        #               (epoch, total_steps))
        #         model.save('latest')
        #         model.save(epoch)
        #
        #     print('End of epoch %d / %d \t Time Taken: %d sec' %
        #           (epoch, self.opt.niter + self.opt.niter_decay, time.time() - epoch_start_time))
        #
        #     if epoch > self.opt.niter:
        #         model.update_learning_rate()



    def seprate_train_val(self):
        random.shuffle(self.data_paths)
        number_val = int(len(self.data_paths) * self.option.val_rate)
        val_data = self.data_paths[:number_val]
        train_data = self.data_paths[number_val:]

        return train_data, val_data

    def create_model(self):
        model = None
        if self.option.model_type == 'GAN':
            from .cycle_gan_model import CycleSEGModel
            model = CycleSEGModel()
            print("model [%s] was created" % (self.option.model_type))
        model.initialize(self.option)
        return model








