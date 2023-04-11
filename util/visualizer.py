import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        now = time.strftime("%c")
        self.metrics_image_dir = os.path.join('results', self.opt.name, self.opt.metrics_image_dir)
        os.makedirs(self.metrics_image_dir, exist_ok=True)

        if self.opt.operation == 'Train':

            self.logs_dir = os.path.join('results', self.opt.name, self.opt.logs_dir)
            os.makedirs(self.logs_dir, exist_ok=True)


            self.img_dir = os.path.join('results', self.opt.name, self.opt.val_iamge_dir)
            os.makedirs(self.img_dir, exist_ok=True)

            self.loss_log = os.path.join(self.logs_dir, 'loss_log.txt')
            self.data_log = os.path.join(self.logs_dir, 'data_log.txt')
            self.model_log = os.path.join(self.logs_dir, 'model_log.txt')



            with open(self.loss_log, "a") as file:
                file.write(f"loss log file for , {self.name}. time: {now}.\n")

            with open(self.data_log, "a") as file:
                file.write(f"data log file for , {self.name}. time: {now}.\n")

            with open(self.model_log, "a") as file:
                file.write(f"model log file for , {self.name}. time: {now}.\n")

        elif self.opt.operation == 'Test':
            self.img_dir = os.path.join('results', self.opt.name, self.opt.test_image_dir)
            os.makedirs(self.img_dir, exist_ok=True)

            self.logs_dir = os.path.join('results', self.opt.name, self.opt.logs_dir)
            os.makedirs(self.logs_dir, exist_ok=True)
            self.test_log = os.path.join(self.logs_dir, 'test_log.txt')

            with open(self.test_log, "a") as file:
                file.write(f"test log file for , {self.name}. time: {now}.\n")



    # |visuals|: dictionary of images to display or save
    def log_model(self,txt):
        now = time.strftime("%c")
        with open(self.model_log, "a") as file:
            file.write( f"add model log time: {now}.\n")
            file.write(f"{txt}\n")

    def log_data(self,txt):
        now = time.strftime("%c")
        with open(self.data_log, "a") as file:
            file.write( f"add data log time: {now}.\n")
            file.write(f"{txt}\n")

    def log_loss(self,txt):
        now = time.strftime("%c")
        with open(self.loss_log, "a") as file:
            file.write( f"add loss log time: {now}.\n")
            file.write(f"{txt}\n")

    def log_test(self,txt):
        now = time.strftime("%c")
        with open(self.test_log, "a") as file:
            file.write( f"add test log time: {now}.\n")
            file.write(f"{txt}\n")

    def plot_metrics(self, metrics, tag):
        plt.figure(1,figsize=(12, 6))
        plt.subplot(1, 1, 1)
        plt.title(tag)
        x = [i + 1 for i in range(len(metrics))]
        y = metrics
        plt.xlabel("epoch")
        plt.plot(x, y)
        path_img = os.path.join(self.metrics_image_dir,f"{tag}.png")
        plt.savefig(path_img)
        plt.close()

    def print_model(self, net, label):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        with open(self.model_log, "a") as file:
            file.write(f"Total number of parameters for {label}: {num_params}.\n")
            file.write(f"Summary of the  {label}: \n")
            print(net, file=file)

    def save_images(self, img, tag, label=False, epoch=0):

        fig, axs = plt.subplots(1, len(self.opt.save_slices), figsize=(15, 5))
        for i in range(len(self.opt.save_slices)):
            n_slice = self.opt.save_slices[i]
            axs[i].set_title(f"{tag} image slice # {n_slice}")
            if not label:
                axs[i].imshow(img[:, :, n_slice], cmap="gray")
            else:
                axs[i].imshow(img[:, :, n_slice])

        folder_path = os.path.join(self.img_dir, str(epoch))
        if os.path.exists(folder_path):
            print("Folder created successfully.")
        else:
            os.makedirs(folder_path, exist_ok=True)

        plt.savefig(os.path.join(folder_path, f"{tag}_image_slices_{self.opt.save_slices}.png"))
        plt.close()




