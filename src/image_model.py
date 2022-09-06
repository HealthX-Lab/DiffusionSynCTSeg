import gin
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import ants
@gin.configurable
class Image:
    def __init__(self, dic_dir_to_images):
        self.dic_dir_to_images = dic_dir_to_images
        self.image_dic = {k: None for k, v in dic_dir_to_images}
        self.read_images()

    def read_images(self):
        for dataset_names , path_to_images in self.dic_dir_to_images:
            self.image_dic[dataset_names] = [(nib.load(img_path)).get_fdata() for img_path in path_to_images]
            print(np.shape(self.image_dic))
    # def read_nii(self):

