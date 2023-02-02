
# import monai.transforms
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByLabelClassesd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)

class MyTransformer:
    def __init__(self, opt,operation):
        self.operation = operation
        self.transformers = opt.transformers[self.operation]
        self.data_transformer_list = []
        self.compose_transformer()
        self.data_transforms = Compose(self.data_transformer_list)

    def compose_transformer(self):

        for dic_transformers in self.transformers:
            transformer_class = eval(dic_transformers['cls'])
            tmp = transformer_class(**dic_transformers['arg'])
            self.data_transformer_list.append(tmp)

    def get_data_transformer(self):
        return self.data_transforms

