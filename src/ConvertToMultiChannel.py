from monai.transforms import MapTransform
import torch
import numpy as np


class ConvertToMultiChannelVentricleClasses(MapTransform):
    """
    Convert labels to multi channels based on ventricle classes
    label 0  foreground
    label 1 3rd ventricle
    label 2 4th ventricle
    label 3 5th ventricle
    label 4 left and right ventricle

    """

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            a0 = torch.where(d[key] == 0, 0, 0)
            a1 = torch.where(d[key] == 50, 1, 0)
            a2 = torch.where(d[key] == 100, 2, 0)
            a3 = torch.where(d[key] == 150, 3, 0)
            a4 = torch.where(d[key] == 200, 4, 0)
            d[key] = torch.sum(torch.stack((a0, a1, a2, a3, a4), dim=0), dim=0)

        return d