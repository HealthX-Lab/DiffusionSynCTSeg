
class Model:
    def __init__(self, opt):
        self.option = opt
        self.create_model()
    def create_model(self):
        if self.option.model_type == 'GAN':
            from .cycle_gan_model import CycleGANModel
            model = CycleGANModel()
        return model


