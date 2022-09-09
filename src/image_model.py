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
    def create_list_images(self,images_with_modality,num_images_in_row,slide):
        images_list = []
        modality_list = []
        for modality, images in images_with_modality.items():
            modality_list.append(modality)
            for image in images[:num_images_in_row]:
                images_list.append(image[:, :, slide])
        return modality_list, images_list
    def show_images(self, slide=90, show_plot=True, save_plot=True, num_images_in_row=3):
        for dataset_name, images_with_modality in self.image_dic.items():
            fig, axs = plt.subplots(len(images_with_modality), num_images_in_row)#(len(images_with_modality), num_images_in_row)
            fig.suptitle(dataset_name)
            modality_list, images_list = self.create_list_images(images_with_modality,num_images_in_row,slide)
            i = 0
            for ax, image in zip(axs.flat, images_list):
                ax.imshow(image)
                if (i%3==0):
                    ax.set_ylabel(modality_list[i//3])
                i += 1


            if save_plot:
                fig.savefig(os.path.join('./plots',f"{dataset_name}-samples.jpg"))
            if show_plot:
                plt.show()









