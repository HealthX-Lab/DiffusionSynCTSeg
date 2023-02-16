import numpy as np
import os
import ntpath
import time
from . import util
from . import html

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name

        if self.opt.operation == 'Train':

            self.logs_dir = os.path.join('results', self.opt.name, self.opt.logs_dir)
            os.makedirs(self.logs_dir, exist_ok=True)

            self.metrics_image_dir = os.path.join('results', self.opt.name, self.opt.metrics_image_dir)
            os.makedirs(self.metrics_image_dir, exist_ok=True)

            self.loss_log = os.path.join(self.logs_dir, 'loss_log.txt')
            self.data_log = os.path.join(self.logs_dir, 'data_log.txt')
            self.model_log = os.path.join(self.logs_dir, 'model_log.txt')

            now = time.strftime("%c")

            with open(self.loss_log, "a") as file:
                file.write(f"loss log file for , {self.name}. time: {now}.\n")

            with open(self.data_log, "a") as file:
                file.write(f"data log file for , {self.name}. time: {now}.\n")

            with open(self.model_log, "a") as file:
                file.write(f"model log file for , {self.name}. time: {now}.\n")

        elif self.opt.operation == 'Test':
            self.img_dir = os.path.join('results', self.opt.name, self.opt.test_iamge_dir)
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
        plt.figure( (12, 6))
        plt.subplot(1, 1, 1)
        plt.title(tag)
        x = [i + 1 for i in range(len(metrics))]
        y = metrics
        plt.xlabel("epoch")
        plt.plot(x, y)
        os.path.join(self.img_dir,f"{tag}.png")
        plt.savefig()
        plt.close()

    def print_model(self, net, label):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        with open(self.model_log, "a") as file:
            file.write(f"Total number of parameters for {label}: {num_params}.\n")
            file.write(f"Summary of the  {label}: \n")
            print(net, file=file)

    def save_images(self,img,tag,label):

        fig, axs = plt.subplots(1,len(self.opt.save_slices), figsize=(15, 5))
        for i in len (self.opt.save_slices):
            n_slice = self.opt.save_slices[i]
            axs[i].set_title(f"{tag} image slice # {n_slice}")
            if not label:
                axs[i].imshow(img[:, :, n_slice], cmap="gray")
            else:
                axs[i].imshow(img[:, :, n_slice])

        plt.savefig(os.path.join(self.img_dir, f"{tag}_image_slices_{self.opt.save_slices}.png"))
        plt.close()





#
#     def display_current_results(self, visuals, epoch):
#         if self.display_id > 0: # show images in the browser
#             if self.display_single_pane_ncols > 0:
#                 h, w = next(iter(visuals.values())).shape[:2]
#                 table_css = """<style>
#     table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
#     table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
# </style>""" % (w, h)
#                 ncols = self.display_single_pane_ncols
#                 title = self.name
#                 label_html = ''
#                 label_html_row = ''
#                 nrows = int(np.ceil(len(visuals.items()) / ncols))
#                 images = []
#                 idx = 0
#                 for label, image_numpy in visuals.items():
#                     label_html_row += '<td>%s</td>' % label
#                     images.append(image_numpy.transpose([2, 0, 1]))
#                     idx += 1
#                     if idx % ncols == 0:
#                         label_html += '<tr>%s</tr>' % label_html_row
#                         label_html_row = ''
#                 white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
#                 while idx % ncols != 0:
#                     images.append(white_image)
#                     label_html_row += '<td></td>'
#                     idx += 1
#                 if label_html_row != '':
#                     label_html += '<tr>%s</tr>' % label_html_row
#                 # pane col = image row
#                 self.vis.images(images, nrow=ncols, win=self.display_id + 1,
#                                 padding=2, opts=dict(title=title + ' images'))
#                 label_html = '<table>%s</table>' % label_html
#                 self.vis.text(table_css + label_html, win = self.display_id + 2,
#                               opts=dict(title=title + ' labels'))
#             else:
#                 idx = 1
#                 for label, image_numpy in visuals.items():
#                     #image_numpy = np.flipud(image_numpy)
#                     self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
#                                        win=self.display_id + idx)
#                     idx += 1
#
#         if self.use_html: # save images to a html file
#             for label, image_numpy in visuals.items():
#                 img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
#                 util.save_image(image_numpy, img_path)
#             # update website
#             webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
#             for n in range(epoch, 0, -1):
#                 webpage.add_header('epoch [%d]' % n)
#                 ims = []
#                 txts = []
#                 links = []
#
#                 for label, image_numpy in visuals.items():
#                     img_path = 'epoch%.3d_%s.png' % (n, label)
#                     ims.append(img_path)
#                     txts.append(label)
#                     links.append(img_path)
#                 webpage.add_images(ims, txts, links, width=self.win_size)
#             webpage.save()
#
#     # errors: dictionary of error labels and values
#     def plot_current_errors(self, epoch, counter_ratio, opt, errors):
#         if not hasattr(self, 'plot_data'):
#             self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
#         self.plot_data['X'].append(epoch + counter_ratio)
#         self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
#         self.vis.line(
#             X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
#             Y=np.array(self.plot_data['Y']),
#             opts={
#                 'title': self.name + ' loss over time',
#                 'legend': self.plot_data['legend'],
#                 'xlabel': 'epoch',
#                 'ylabel': 'loss'},
#             win=self.display_id)
#
#     # errors: same format as |errors| of plotCurrentErrors
#     def print_current_errors(self, epoch, i, errors, t):
#         message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
#         for k, v in errors.items():
#             message += '%s: %.3f ' % (k, v)
#
#         print(message)
#         with open(self.log_name, "a") as log_file:
#             log_file.write('%s\n' % message)
#
#     # save image to the disk
#     def save_images(self, webpage, visuals, image_path):
#         image_dir = webpage.get_image_dir()
#         short_path = ntpath.basename(image_path[0])
#         name = os.path.splitext(short_path)[0]
#
#         webpage.add_header(name)
#         ims = []
#         txts = []
#         links = []
#
#         for label, image_numpy in visuals.items():
#             image_name = '%s_%s.png' % (name, label)
#             save_path = os.path.join(image_dir, image_name)
#             util.save_image(image_numpy, save_path)
#
#             ims.append(image_name)
#             txts.append(label)
#             links.append(image_name)
#         webpage.add_images(ims, txts, links, width=self.win_size)
#
#     def mkdir(self,path):
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#     # save image to the disk
#     def save_images_to_dir(self, image_dir, visuals, image_path):
#         short_path = ntpath.basename(image_path[0])
#         name = os.path.splitext(short_path)[0]
#         full_path_strs = image_path[0].split('/')
#
#         save_dir = os.path.join(image_dir, 'img_fake_only', full_path_strs[-3], full_path_strs[-2])
#         self.mkdir(save_dir)
#
#         label = 'fake_B'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir,image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#         save_dir = os.path.join(image_dir, 'img_all', full_path_strs[-3], full_path_strs[-2])
#         self.mkdir(save_dir)
#
#         label = 'fake_B'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir, image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#         label = 'real_A'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir,image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#         label = 'real_B'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir,image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#         label = 'fake_A'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir,image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#         label = 'rec_A'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir,image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#         label = 'rec_B'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir,image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#
#
#
#     def save_seg_images_to_dir(self, image_dir, visuals, image_path):
#         short_path = ntpath.basename(image_path[0])
#         name = os.path.splitext(short_path)[0]
#         full_path_strs = image_path[0].split('/')
#
#         save_dir = os.path.join(image_dir, 'img_fake_only', full_path_strs[-3], full_path_strs[-2])
#         self.mkdir(save_dir)
#
#         label = 'fake_B'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir,image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#         save_dir = os.path.join(image_dir, 'img_all', full_path_strs[-3], full_path_strs[-2])
#         self.mkdir(save_dir)
#
#         label = 'fake_B'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir, image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)
#
#         label = 'real_A'
#         image_numpy = visuals[label]
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(save_dir,image_name)
#         if not os.path.exists(save_path):
#             util.save_image(image_numpy, save_path)