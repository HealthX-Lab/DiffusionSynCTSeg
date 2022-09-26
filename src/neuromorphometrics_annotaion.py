from src.dataset_model import Dataset
import os
class NeuroMorphometricAnnotatedDataset(Dataset):
    def __init__(self, name, directory, attributes):
        Dataset.__init__(self, name, directory, attributes)
        self.patients_directory = [os.path.join(self.path_to_image_directory[0], image_address)
                                  for image_address in os.listdir(self.path_to_image_directory[0])]
        self.create_path_to_images_per_modality()
        self.create_path_to_images_per_patient()

    def find_modalities(self,patient):
        tmp_dict = {key: [] for key in self.type}
        for image_address in os.listdir(patient):
            try:
                if image_address.find('seg') != -1:
                    tmp_dict['GT'].append(os.path.join(patient, image_address))
                elif image_address.find('.nii') != -1 and image_address.find('MR') != -1:
                    tmp_dict['MR'].append(os.path.join(patient, image_address))

            except ValueError:
                print("Oops!  That modality is not supported")
        return tmp_dict

    def create_path_to_images_per_modality(self):
        for patient in self.patients_directory:
            modalities_dict = self.find_modalities(patient)
            self.path_to_images_per_modality['MR'].append(modalities_dict['MR'][0])
            self.path_to_images_per_modality['GT'].append(modalities_dict['GT'][0])


    def create_path_to_images_per_patient(self):
        for patient in self.patients_directory:
            self.path_to_images_per_patient[patient.split('/')[-1]] = self.find_modalities(patient)






