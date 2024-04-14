from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    min_val = np.min(image_numpy)
    max_val = np.max(image_numpy)

    print(f"Minimum value: {min_val}  shape:{np.shape(image_numpy)}")
    print(f"Maximum value: {max_val}")
    # image_numpy = (image_numpy- min_val) / (max_val - min_val)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0

    

    return image_numpy.astype(imtype)


def tensor2im_real(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    min_val = np.min(image_numpy)
    max_val = np.max(image_numpy)

    print(f"Minimum value in real: {min_val}  shape:{np.shape(image_numpy)}")
    print(f"Maximum value in real: {max_val}")
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))*255

    print(f"Minimum value in real after 255: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
    print(f"Maximum value in real after 255: {np.max(image_numpy)}")
    #
    uint_image = image_numpy.astype(imtype)
    #
    print(f"Minimum value in real after uint: {np.min(uint_image)}  shape:{np.shape(uint_image)}")
    print(f"Maximum value in real after uint: {np.max(uint_image)}")


    return image_numpy.astype(imtype)

def tensor2seg(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    min_val = np.min(image_numpy)
    max_val = np.max(image_numpy)

    print(f" seg Minimum value: {min_val}  shape:{np.shape(image_numpy)}")
    print(f"seg Maximum value: {max_val}")
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) *100
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    print(f"Minimum value in util save image: {np.min(image_numpy)}  shape:{np.shape(image_numpy)}")
    print(f"Maximum value util save image: {np.max(image_numpy)}")


    if (len(image_numpy.shape)>2):
        image_pil = Image.fromarray(image_numpy[:,:,0])
    else:
        image_pil = Image.fromarray(image_numpy)

    print(f"Minimum value in util save image after form array: {np.min(image_pil)}  shape:{np.shape(image_pil)}")
    print(f"Maximum value util save image after form array: {np.max(image_pil)}")
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
