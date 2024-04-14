import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass




def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        print(' I am in resize')
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        print(' I am in crop')
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        print(' I am in scale')
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        print(' I am in scale and crop')
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'yh_test_resize':
        print(' I am in resize')
        osize = [opt.fineSize, opt.fineSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    # elif opt.resize_or_crop == 'resize':
    #     osize = [opt.loadSize, opt.loadSize]
    #     transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    # elif opt.resize_or_crop == 'random_crop':
    #     transform_list.append(random_crop_yh.randomcrop_yh(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        print(' I am in flip')
        transform_list.append(transforms.RandomHorizontalFlip())

    # transform_list += [transforms.ToTensor(),
    #                    transforms.Normalize((0.5),
    #                                         (0.5))]
    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
