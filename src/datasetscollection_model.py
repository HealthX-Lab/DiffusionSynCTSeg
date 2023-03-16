
import os



class DatasetsCollection:
    def __init__(self,  opt):
        self.directory = opt.dataroot
        self.operation = opt.operation if opt.operation != "check_image" else "Train"
        self.attributes = opt.attributes[self.operation]
        self.data_dic = []
        self.names = [name for name in self.attributes if self.attributes[name]['flag']]
        self.image_folder_path = {name: [] for name in self.attributes if self.attributes[name]['flag']}
        self.path_to_images = {name: None for name in self.attributes if self.attributes[name]['flag']}
        self.folder_names = {name: self.attributes[name]['folder_name'] for name in self.attributes if self.attributes[name]['flag']}
        self.types = {name: self.attributes[name]['type'] for name in self.attributes if self.attributes[name]['flag']}
        self.input_types =opt.input_type[self.operation]
        self.load_datasets_paths()




    def load_datasets_paths(self):
        self.create_paths_to_dataset()
        self.set_image_paths()
        self.aggrigate_dataset_paths()



    def create_paths_to_dataset(self):
        for dataset_name in self.names:
            for type,folder_path in  zip(self.types[dataset_name], self.attributes[dataset_name]['folder_paths']):
                self.image_folder_path[dataset_name].append(os.path.join(self.directory,self.folder_names[dataset_name][0],type,folder_path))

    def set_image_paths(self):
        for dataset_name in self.names:
            self.path_to_images[dataset_name] = {type:[] for type in self.types[dataset_name]}
            for i in range (0,len (self.image_folder_path[dataset_name])):
                type = self.types[dataset_name][i]
                for file in sorted(os.listdir(self.image_folder_path[dataset_name][i])):
                     self.path_to_images[dataset_name][type].append(os.path.join(self.image_folder_path[dataset_name][i],file))





    def aggrigate_dataset_paths(self):
        self.path_per_modality = {modality:[] for modality in self.input_types}
        for dataset_name, value in self.path_to_images.items():
            for type, paths in value.items():
                self.path_per_modality[type].extend(value[type])


    def get_data_dics(self):
        return self.data_dic

    def over_samples(self):

        min_len = min(len(self.path_per_modality['MRI']), len(self.path_per_modality['CT']))
        max_len = max(len(self.path_per_modality['MRI']), len(self.path_per_modality['CT']))
        i = 0
        if min_len == len(self.path_per_modality['MRI']):
            for j in range(0, max_len):
                if i == min_len:
                    i = 0
                self.data_dic.append({'MRI': self.path_per_modality['MRI'][i], 'label': self.path_per_modality['label'][i],
                                 'CT': self.path_per_modality['CT'][j]})
                i += 1
        elif min_len == len(self.path_per_modality['CT']):
            for j in range(0, max_len):
                if i == min_len:
                    i = 0
                self.data_dic.append({'MRI': self.path_per_modality['MRI'][j], 'label': self.path_per_modality['label'][j],
                                       'CT': self.path_per_modality['CT'][i]})
                i += 1



    def under_samples(self):

        min_len = min(len(self.path_per_modality['MRI']),len(self.path_per_modality['CT']))
        for i in range (0,min_len):
             self.data_dic.append({'MRI':self.path_per_modality['MRI'][i],'label':self.path_per_modality['label'][i], 'CT':self.path_per_modality['CT'][i]})

    def test_samples(self):
        for i in range (0,len(self.path_per_modality['CT'])):
             self.data_dic.append({'label':self.path_per_modality['label'][i], 'CT':self.path_per_modality['CT'][i]})


    def paired_samples(self):

        for i in range(0, len(self.path_per_modality[self.input_types[0]])):
            self.data_dic.append({self.input_types[0]: self.path_per_modality[self.input_types[0]][i],
                             self.input_types[1]: self.path_per_modality[self.input_types[1]][i]})







