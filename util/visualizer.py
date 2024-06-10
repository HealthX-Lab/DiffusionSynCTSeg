import numpy as np
import os
import ntpath
import time
import util.util as util
# from . import html

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = False
        self.win_size = opt.display_winsize
        self.name = opt.name

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save


    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        # image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        # webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        # webpage.add_images(ims, txts, links, width=self.win_size)

    def mkdir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_images_to_dir_uncertainty(self, image_dir, image_dict, image_path):
        print('image_path: ', image_path[0])
        short_path = ntpath.basename(image_path[0])
        print('short_path: ',short_path)
        name = os.path.splitext(short_path)[0]
        full_path_strs = image_path[0].split('/')

        # save_dir = os.path.join(image_dir, 'img_fake_only', full_path_strs[-3], full_path_strs[-2])
        # self.mkdir(save_dir)
        save_dir = os.path.join(image_dir, 'img_all', full_path_strs[-3], full_path_strs[-2])
        self.mkdir(save_dir)
        #*********************************
        label = 'real_A'
        image_numpy = image_dict[label]
        print(f"Minimum value in visualisation A: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
        print(f"Maximum value in visualisation A: {np.max(image_numpy)}")

        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        label = 'real_B'
        image_numpy = image_dict[label]
        print(f"Minimum value in visualisation B: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
        print(f"Maximum value invisualisation B: {np.max(image_numpy)}")
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        label = 'input_seg'
        image_numpy = image_dict[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        label = 'seg_gamma1'
        image_numpy = image_dict[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        label = 'seg_gamma2'
        image_numpy = image_dict[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        # *********************************
        for key, value in image_dict['Visuals'].items():
            print(key,'&&&&&**&*&')
            for i in range(0,1) :#self.opt.num_samples_uncertainty
                label = f'{key}_{i}'
                image_numpy = value[i]
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(save_dir, image_name)
                # if not os.path.exists(save_path):
                util.save_image(image_numpy, save_path)

            if 'seg' not in key:


                label = f'{key}_uncertainty_map'
                uncertainty_map = image_dict['Uncertainty_Map']
                image_numpy = uncertainty_map[key]
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(save_dir, image_name)
                # if not os.path.exists(save_path):
                util.save_map(image_numpy, save_path)

            if  key =='seg_real':

                label = f'{key}_heatmap'
                heatmap = image_dict['Heatmap']
                image_numpy = heatmap[key]
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(save_dir, image_name)
                # if not os.path.exists(save_path):
                util.save_image(image_numpy, save_path)

                label = f'{key}_confidence_map'
                confidence_map = image_dict['Confidence_Map']
                image_numpy = confidence_map[key]
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(save_dir, image_name)
                # if not os.path.exists(save_path):
                # util.save_image(image_numpy, save_path)
                util.save_map(image_numpy, save_path)

                #
                label = f'{key}_entropy_map'
                entropy_map = image_dict['Entropy_Map']
                image_numpy = entropy_map[key]
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(save_dir, image_name)
                # if not os.path.exists(save_path):
                # util.save_image(image_numpy, save_path)
                util.save_map(image_numpy, save_path)
                if key == 'fake_A' or key == 'fake_B' or key == 'seg_real':
                    label = f'{key}_uncertainty_map'
                    uncertainty_map = image_dict['Uncertainty_Map']
                    image_numpy = uncertainty_map[key]
                    image_name = '%s_%s.png' % (name, label)
                    save_path = os.path.join(save_dir, image_name)
                    # if not os.path.exists(save_path):
                    util.save_map(image_numpy, save_path)







    # save image to the disk
    def save_images_to_dir(self, image_dir, visuals, image_paths_A, image_paths_B, image_paths_seg):
        print('I am in save image')
        short_path = ntpath.basename(image_paths_A[0])
        # name_A = os.path.splitext(short_path)[0]
        name = os.path.splitext(short_path)[0]
        full_path_strs = image_paths_A[0].split('/')

        save_dir = os.path.join(image_dir)
        self.mkdir(save_dir)

        label = 'fake_B'
        image_numpy = visuals[label]
        # image_name = '%s_%s.png' % (name_A, label)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        print(save_path)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)
        # #
        label = 'real_A'
        image_numpy = visuals[label]
        print(f"Minimum value in visualisation A: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
        print(f"Maximum value in visualisation A: {np.max(image_numpy)}")

        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)
        #
        label = 'real_B'
        image_numpy = visuals[label]
        print(f"Minimum value in visualisation B: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
        print(f"Maximum value invisualisation B: {np.max(image_numpy)}")
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)
        #
        label = 'fake_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        label = 'rec_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        label = 'rec_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        label = 'fake_seg'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        #if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)



        label = 'real_seg'
        image_numpy = visuals[label]
        # image_name = '%s_%s.png' % (name_B, label)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        print(save_path)
        #if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)
        #
        label = 'input_seg'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        #if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        print('I am out of visoualizer')




    def save_seg_images_to_dir(self, image_dir, visuals, image_path):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        full_path_strs = image_path[0].split('/')

        save_dir = os.path.join(image_dir, 'img_fake_only', full_path_strs[-3], full_path_strs[-2])
        self.mkdir(save_dir)




        label = 'fake_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        save_dir = os.path.join(image_dir, 'img_all', full_path_strs[-3], full_path_strs[-2])
        self.mkdir(save_dir)

        if self.opt.uncertainty:
            label = 'entropy'
            image_numpy = visuals[label]
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(save_dir, image_name)
            if not os.path.exists(save_path):
                util.save_image(image_numpy, save_path)

        label = 'fake_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        label = 'real_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)


    def save_cycle_gan_images_to_dir(self, image_dir, visuals, image_paths_A, image_paths_B):
        print('I am in save image')
        short_path = ntpath.basename(image_paths_A[0])
        # name_A = os.path.splitext(short_path)[0]
        name = os.path.splitext(short_path)[0]
        full_path_strs = image_paths_A[0].split('/')


        save_dir = os.path.join(image_dir)
        self.mkdir(save_dir)

        label = 'fake_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        print(save_path)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)
        #
        label = 'real_A'
        image_numpy = visuals[label]
        print(f"Minimum value in visualisation A: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
        print(f"Maximum value in visualisation A: {np.max(image_numpy)}")

        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)
        #
        label = 'real_B'
        image_numpy = visuals[label]
        print(f"Minimum value in visualisation B: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
        print(f"Maximum value invisualisation B: {np.max(image_numpy)}")
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)
        #
        label = 'fake_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        label = 'rec_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        label = 'rec_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)
        #


        print('I am out of visoualizer')

    def save_cycle_gan_images_to_dir_uncertainty(self, cycle_output_dir, visuals, img_path):
        print('hello save_cycle_gan_images_to_dir_uncertainty function')



    def save_val_images_to_dir(self, image_dir, visuals, epoch,val_num):
        print('I am in save image')
        name = f'{epoch}_{val_num}'

        save_dir = os.path.join(image_dir)
        self.mkdir(save_dir)

        label = 'fake_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        print(save_path)
        util.save_image(image_numpy, save_path)

        label = 'fake_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        util.save_image(image_numpy, save_path)

        label = 'rec_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        util.save_image(image_numpy, save_path)

        label = 'rec_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        util.save_image(image_numpy, save_path)

        label = 'fake_seg'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        util.save_image(image_numpy, save_path)


        label = 'real_seg'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        print(save_path)
        util.save_image(image_numpy, save_path)

        label = 'input_seg'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        # if not os.path.exists(save_path):
        util.save_image(image_numpy, save_path)

        print('I am out of visoualizer')
