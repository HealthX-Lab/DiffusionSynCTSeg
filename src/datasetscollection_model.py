import gin
import os
import sys
from src.NonContrastDataLoader_model import NonContrastDataLoader
from src.iDBCERMEPLoader_model import iDBCERMEPLoader
from src.neuromorphometrics_annotaion import  NeuroMorphometricAnnotatedDataset

@gin.configurable
class DatasetsCollection:
    def __init__(self, dir_data, attributes):
        self.directory = dir_data
        self.attributes = attributes
        self.names = [names for names in self.attributes]
        self.path = {names: None for names in self.attributes}
        self.load_datasets()


    def load_datasets(self):
        for name in self.names:
            dataset_class = eval(self.attributes[name]['class'])
            class_obj = dataset_class(name, self.directory, self.attributes[name])
            self.path[name] = class_obj.get_path_to_images()

    def get_paths(self):
        return self.path





