import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from src.datasetscollection_model import DatasetsCollection
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.data import CacheDataset, DataLoader
import random
from src.transformer_model import MyTransformer
import random
class TestChain:
    def __init__(self, opt, paths):
        self.test_data = random.shuffle(self.paths)
        self.data_paths = paths
        self.option = opt
        self.confusion_matrix_list = {i: [] for i in self.option.ConfusionMatrixMetric}
        self.visualizer = self.Visualizer(self.option)


    def build_test_transformers(self):
        transformer_obj_test= MyTransformer(self.option,'Test')
        test_data_transformes = transformer_obj_test.get_data_transformer()
        test_ds = CacheDataset(data=self.test_data,
                                transform=test_data_transformes,
                                cache_rate=self.option.cache_rate,
                                num_workers=self.option.dataLoader_num_workers)
        self.test_loader = DataLoader(test_ds, batch_size=self.option.batch_size)
        self.test_size = len(self.test_loader)

    def test_model(self):
        self.model.eval()
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                 net_names = self.model.get_networks_name()
                 nets = self.model.get_networks()
                 net_dic = {name: net for name, net in zip (net_names,nets)}

                 G_A = net_dic['G_A']
                 G_B = net_dic['G_B']
                 G_seg = net_dic['G_seg']

                 fake_B = sliding_window_inference(
                     test_data["MRI"].to(device), self.option.roi_size, self.option.sw_batch_size, G_A
                 )

                 fake_A = sliding_window_inference(
                     test_data["CT"].to(device), self.option.roi_size, self.option.sw_batch_size, G_B
                 )

                 rec_A = sliding_window_inference(
                     fake_B.to(device), self.option.roi_size, self.option.sw_batch_size, G_B
                 )

                 rec_B = sliding_window_inference(
                     fake_A.to(device), self.option.roi_size, self.option.sw_batch_size, G_A
                 )

                 seg_real = sliding_window_inference(
                     test_data["CT"].to(device), self.option.roi_size, self.option.sw_batch_size, G_seg
                 )

                 seg_fake = sliding_window_inference(
                     fake_B.to(device), self.option.roi_size, self.option.sw_batch_size, G_seg
                 )
                 post_pred_seg = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=self.option.num_classes)])
                 post_pred_images = Compose([EnsureType()])
                 seg_real = [post_pred_seg(i) for i in decollate_batch(seg_real)]
                 self.visualizer(seg_real,'segmentation from real CT',True)

                 seg_fake = [post_pred_seg(i) for i in decollate_batch(seg_fake)]
                 self.visualizer(seg_fake, 'segmentation from synthesis CT', True)

                 fake_A = [post_pred_images(i) for i in decollate_batch(fake_A)]
                 self.visualizer(fake_A, 'G_B(CT)', False)

                 fake_B = [post_pred_images(i) for i in decollate_batch(fake_B)]
                 self.visualizer(fake_B, 'G_A(MRI)', False)

                 rec_A = [post_pred_images(i) for i in decollate_batch(rec_A)]
                 self.visualizer(rec_A, 'G_A(G_B(MRI))', False)

                 rec_B = [post_pred_images(i) for i in decollate_batch(rec_B)]
                 self.visualizer(rec_B, 'G_B(G_A(CT))', False)


                 for i in len(self.option.ConfusionMatrixMetric):
                     self.confusion_matrix_list[self.option.ConfusionMatrixMetric[i]].append(
                         confusion_matrix.aggregate()[i].item())

                 for i in len(self.option.ConfusionMatrixMetric):
                     txt = f" {self.option.ConfusionMatrixMetric[i]}: " \
                           f"{sum(self.confusion_matrix_list[self.option.ConfusionMatrixMetric[i]])}\n "
                     self.visualizer.log_test(txt)




        print('all dice=', sum_dice / 15)
        print('all sensitivity=', sum_sensitivity / 15)
        print('all precision=', sum_precision / 15)
        print('all specificity=', sum_specificity / 15)

    def build_chain(self):
        self.build_transformers()
        self.model = self.create_model()
        self.test_model()











