import gin
import os
@gin.configurable
class Dataset():
    def __init__(self, dir_data, dataset_names, paths):
        self.dir_data = dir_data
        self.dataset_names = dataset_names
        self.paths = paths
        self.path_to_datasets_images_dic = {key: [] for key in self.dataset_names}
        self.path_to_image_directory = None
        self.create_path_to_image_directory()
        self.create_path_to_images()
        self.path_to_datasets_images_dic


    def print_address(self):
        print(self.dir_data)

    def create_path_to_image_directory(self):
        self.path_to_image_directory= [os.path.join(self.dir_data, path)  for  path in self.paths]
        print(self.path_to_image_directory)

    def create_path_to_images(self):

        for path, data_name in zip(self.path_to_image_directory, self.dataset_names):
            self.path_to_datasets_images_dic[data_name] = [os.path.join(path, image_address) for image_address in os.listdir(path)]
        return self.path_to_datasets_images_dic




