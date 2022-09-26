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
    def __init__(self, flag_train, **kwargs):
        # kwargs = kwargs['train'] if flag_train else kwargs['test']

        if flag_train:
            kwargs = kwargs['train']
        # train_transforms = Compose(
        #     [
        self.transformers = kwargs['transformers']
        self.compose_transformer()
        a=2
    def compose_transformer(self):
        a = self.transformers
        for dic_transformers in self.transformers:
            transformer_class = eval(dic_transformers['cls'])
            transformer_obj = transformer_class(**dic_transformers['arg'])

