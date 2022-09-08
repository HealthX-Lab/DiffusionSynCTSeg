import gin
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import ants

class Image:
    def __init__(self, image_path):
        self.image_paths = image_path
        self.image_dic = {k: {} for k in image_path}
        self.read_images_per_modality()

    def read_images_per_modality(self):
        for dataset_name, path_to_images in self.image_paths.items():

            for modality, img_paths in self.image_paths[dataset_name]['path_per_modality'].items():
                self.image_dic[dataset_name][modality] = [(nib.load(path)).get_fdata() for path in img_paths]

    def get_images(self):
        return self.image_dic




