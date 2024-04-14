import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        print('save file name ****** ',save_filename)
        save_path = os.path.join(self.save_dir, save_filename)
        # if 'Seg' in network_label:
        #     load_dir = '/home/rtm/scratch/model_outputs/checkpoints/final2D/just_seg'
        #     save_path = os.path.join(load_dir, save_filename)

        print('save path ',save_path)
        # network.load_state_dict(torch.load(save_path),strict=False)
        state_dict = torch.load(save_path)
        missings = []
        unexpecteds = []
        # if self.opt.uncertainty:
        #     missings = ["model.7.weight", "model.7.bias",
        #                 "model.10.conv_block.1.weight", "model.10.conv_block.1.bias",
        #                 "model.10.conv_block.5.weight", "model.10.conv_block.5.bias",
        #                 "model.11.conv_block.1.weight", "model.11.conv_block.1.bias",
        #                 "model.11.conv_block.5.weight", "model.11.conv_block.5.bias",
        #                 "model.12.conv_block.5.weight", "model.12.conv_block.5.bias",
        #                 "model.13.conv_block.5.weight", "model.13.conv_block.5.bias",
        #                 "model.14.conv_block.5.weight", "model.14.conv_block.5.bias",
        #                 "model.15.conv_block.5.weight", "model.15.conv_block.5.bias",
        #                 "model.16.conv_block.5.weight", "model.16.conv_block.5.bias",
        #                 "model.17.conv_block.5.weight", "model.17.conv_block.5.bias",
        #                 "model.18.conv_block.5.weight", "model.18.conv_block.5.bias",
        #                 "model.19.weight", "model.19.bias",
        #                 "model.22.weight", "model.22.bias",
        #                 "model.26.weight", "model.26.bias"]
        #
        #     unexpecteds = ["model.8.weight", "model.8.bias",
        #                    "model.12.conv_block.6.weight", "model.12.conv_block.6.bias",
        #                    "model.13.conv_block.6.weight", "model.13.conv_block.6.bias",
        #                    "model.14.conv_block.6.weight", "model.14.conv_block.6.bias",
        #                    "model.15.conv_block.6.weight", "model.15.conv_block.6.bias",
        #                    "model.16.conv_block.6.weight", "model.16.conv_block.6.bias",
        #                    "model.17.conv_block.6.weight", "model.17.conv_block.6.bias",
        #                    "model.18.conv_block.6.weight", "model.18.conv_block.6.bias",
        #                    "model.19.conv_block.1.weight", "model.19.conv_block.1.bias",
        #                    "model.19.conv_block.6.weight", "model.19.conv_block.6.bias",
        #                    "model.20.conv_block.1.weight", "model.20.conv_block.1.bias",
        #                    "model.20.conv_block.6.weight", "model.20.conv_block.6.bias",
        #                    "model.21.weight", "model.21.bias",
        #                    "model.25.weight", "model.25.bias",
        #                    "model.30.weight", "model.30.bias"]
        #
        #
        # else:
        #
        # # begin fix
        #     missings = ["model.10.conv_block.5.weight", "model.10.conv_block.5.bias",
        #                 "model.11.conv_block.5.weight", "model.11.conv_block.5.bias",
        #                 "model.12.conv_block.5.weight", "model.12.conv_block.5.bias",
        #                 "model.13.conv_block.5.weight", "model.13.conv_block.5.bias",
        #                 "model.14.conv_block.5.weight", "model.14.conv_block.5.bias",
        #                 "model.15.conv_block.5.weight", "model.15.conv_block.5.bias",
        #                 "model.16.conv_block.5.weight", "model.16.conv_block.5.bias",
        #                 "model.17.conv_block.5.weight", "model.17.conv_block.5.bias",
        #                 "model.18.conv_block.5.weight", "model.18.conv_block.5.bias"]
        #
        #     unexpecteds = [ "model.10.conv_block.6.weight", "model.10.conv_block.6.bias",
        #                     "model.11.conv_block.6.weight", "model.11.conv_block.6.bias",
        #                     "model.12.conv_block.6.weight", "model.12.conv_block.6.bias",
        #                     "model.13.conv_block.6.weight", "model.13.conv_block.6.bias",
        #                     "model.14.conv_block.6.weight", "model.14.conv_block.6.bias",
        #                     "model.15.conv_block.6.weight", "model.15.conv_block.6.bias",
        #                     "model.16.conv_block.6.weight", "model.16.conv_block.6.bias",
        #                     "model.17.conv_block.6.weight", "model.17.conv_block.6.bias",
        #                     "model.18.conv_block.6.weight", "model.18.conv_block.6.bias"]




        # for i in range(len(missings)):
        #     state_dict[missings[i]] = state_dict.pop(unexpecteds[i])
        # patch InstanceNorm checkpoints prior to 0.4


        network.load_state_dict(state_dict)

    # def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
    #     key = keys[i]
    #     if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
    #         if module.__class__.__name__.startswith('InstanceNorm') and \
    #                 (key == 'running_mean' or key == 'running_var'):
    #             if getattr(module, key) is None:
    #                 state_dict.pop('.'.join(keys))
    #         if module.__class__.__name__.startswith('InstanceNorm') and \
    #                 (key == 'num_batches_tracked'):
    #             state_dict.pop('.'.join(keys))
    #     else:
    #         self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        pass
    def add_poisson_noise(self, image_tensor):
        # Print the min and max before applying noise
        # print(f"Before noise - Min: {image_tensor.min()}, Max: {image_tensor.max()}")

        # Rescale from [-1, 1] to [0, 1] for Poisson noise
        lambda_tensor = (image_tensor + 1) / 2 * 255  # Scale to [0, 255] for Poisson
        lambda_tensor = torch.clamp(lambda_tensor, min=0)  # Ensure no negative values

        # Apply Poisson noise using torch.poisson
        noisy_tensor = torch.poisson(lambda_tensor)

        # Scale back down to [0, 1] and then to [-1, 1]
        noisy_tensor = noisy_tensor / 255 * 2 - 1

        # Print the min and max after applying noise
        # print(f"After noise - Min: {noisy_tensor.min()}, Max: {noisy_tensor.max()}")

        return noisy_tensor
