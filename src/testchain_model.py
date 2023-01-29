import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from src.datasetscollection_model import DatasetsCollection
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.data import CacheDataset, DataLoader
import random
from src.transformer_model import MyTransformer
class TestChain:
    def __init__(self, opt, paths):
        self.data = paths
        self.option = opt
        self.build_chain()


    def build_chain(self):
        # self.fill_data_dictionary()
        transformer_obj = MyTransformer(self.option)
        data_transformes= transformer_obj.get_data_transformer()

        test_ds = CacheDataset(data=self.data,
                                transform=data_transformes,
                                cache_rate=self.option.cache_rate,
                                num_workers=self.option.dataLoader_num_workers)
        self.test_loader = DataLoader(test_ds, batch_size= self.option.batch_size, shuffle=True)












