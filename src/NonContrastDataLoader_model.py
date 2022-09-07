from src.dataset_model import Dataset
import os
class NonContrastDataLoader(Dataset):
    def __init__(self, name, directory, attributes):
        Dataset.__init__(self, name, directory, attributes)
        self.create_path_to_images_per_modality()
        self.create_path_to_images_per_patient()

    def create_path_to_images_per_modality(self):
        self.path_to_images_per_modality['CT'] = [os.path.join(self.path_to_image_directory[0], image_address)
                                                  for image_address in os.listdir(self.path_to_image_directory[0])]

    def create_path_to_images_per_patient(self):
        patient_names = [name.replace('.nii.gz','') for name in os.listdir(self.path_to_image_directory[0])]
        for name in os.listdir(self.path_to_image_directory[0]):
            key = name.replace('.nii.gz','')
            self.path_to_images_per_patient[key] = os.path.join(self.path_to_image_directory[0], name)




