import gin
import os
import gin
from abc import ABC, abstractmethod
@gin.configurable
class Dataset(ABC):
    def __init__(self, name, directory, attribute):
        self.directory = directory
        self.name = name
        self.path = attribute['path']
        self.type = attribute['type']
        self.path_to_images_per_modality = {key: [] for key in self.type}
        self.path_to_image_directory = None
        self.path_to_images_per_patient = {}
        self.create_path_to_image_directory()

    def create_path_to_image_directory(self):
        self.path_to_image_directory = [os.path.join(self.directory, path) for path in self.path]

    @abstractmethod
    def create_path_to_images_per_modality(self):
        ...

    @abstractmethod
    def create_path_to_images_per_patient(self):
        ...

    def get_path_to_images(self):
        return self.path_to_images_per_modality




