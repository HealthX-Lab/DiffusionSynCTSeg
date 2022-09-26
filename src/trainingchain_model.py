import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from src.datasetscollection_model import DatasetsCollection
from src.transformer_model import MyTransformer
import gin
@gin.configurable
class TrainingChain:
    def __init__(self, attribute):
        self.train_datasets_obj = DatasetsCollection(attribute)
        self.image_paths = self.train_datasets_obj.get_paths()
        self.dataset_names = self.train_datasets_obj.get_names()
        self.data_dicts = [
            {"image": [], "label": []}
        ]
        self.build_chain()

    def build_chain(self):
        self.fill_data_dictionary()
        self.transformer_obj = MyTransformer(flag_train=1)
        self.data_transformer = self.transformer_obj.get_data_transformer()


    def fill_data_dictionary(self):
        image_name = []
        label_name = []
        self.data_dicts = [{'image': [], 'label': []}]
        for dataset_name in self.dataset_names:
            len_data = len(self.image_paths[dataset_name]['path_per_modality']['GT'])
            for i in range(0, len_data):
                image_name.append(self.image_paths[dataset_name]['path_per_modality']['MR'][i])
                label_name.append(self.image_paths[dataset_name]['path_per_modality']['GT'][i])
        self.data_dicts[0]['image'] = image_name
        self.data_dicts[0]['label'] = label_name




