from src.dataset_model import Dataset
import os
class iDBCERMEPLoader(Dataset):
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
                if image_address.find('ct') != -1:
                    tmp_dict['CT'].append(os.path.join(patient, image_address))
                elif image_address.find('FLAIR') != -1:
                    tmp_dict['FLAIR'].append(os.path.join(patient, image_address))
                elif image_address.find('T1w') != -1:
                    tmp_dict['T1W'].append(os.path.join(patient, image_address))
            except ValueError:
                print("Oops!  That modality is not supported")
        return tmp_dict

    def create_path_to_images_per_modality(self):
        for patient in self.patients_directory:
            modalities_dict = self.find_modalities(patient)
            self.path_to_images_per_modality['CT'].append(modalities_dict['CT'][0])
            self.path_to_images_per_modality['FLAIR'].append(modalities_dict['FLAIR'][0])
            self.path_to_images_per_modality['T1W'].append(modalities_dict['T1W'][0])

    def create_path_to_images_per_patient(self):
        for patient in self.patients_directory:
            self.path_to_images_per_patient[patient.split('/')[-1]] = self.find_modalities(patient)






