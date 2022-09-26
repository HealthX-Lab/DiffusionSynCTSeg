import gin
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
@gin.configurable
class MyTransformer:
    def __init__(self, flag_train, transformers):
        transformers = transformers['train'] if flag_train else transformers['test']
        self.transformers = transformers
        self.data_transformer_list = []
        self.compose_transformer()
        self.data_transforms = monai.transforms.Compose(self.data_transformer_list)

    def compose_transformer(self):
        a = self.transformers
        for dic_transformers in self.transformers:
            transformer_class = eval(dic_transformers['cls'])
            a = transformer_class(**dic_transformers['arg'])
            self.data_transformer_list.append(a)

    def get_data_transformer(self):
        return self.data_transforms

