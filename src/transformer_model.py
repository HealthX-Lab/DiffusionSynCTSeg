
import monai.transforms
from monai import *
# (
#     Transform,
#     AsDiscrete,
#     AsDiscreted,
#     EnsureChannelFirstd,
#     Compose,
#     CropForegroundd,
#     LoadImaged,
#     Orientationd,
#     RandCropByPosNegLabeld,
#     ScaleIntensityRanged,
#     Spacingd,
#     EnsureTyped,
#     EnsureType,
#     Invertd,
#     MapTransform,
#     Randomizable,
# )

class MyTransformer:
    def __init__(self, opt):
        self.operation = opt.operation
        self.transformers = opt.transformers[self.operation]
        self.data_transformer_list = []
        self.compose_transformer()
        self.data_transforms = monai.transforms.Compose(self.data_transformer_list)

    def compose_transformer(self):

        for dic_transformers in self.transformers:
            transformer_class = eval(dic_transformers['cls'])
            tmp = transformer_class(**dic_transformers['arg'])
            self.data_transformer_list.append(tmp)

    def get_data_transformer(self):
        return self.data_transforms

