from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import csv
from collections import OrderedDict
import h5py


def save_image_array(image_array, path, filename):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    if image_array is None:
        raise ValueError("image_array is None, cannot save to HDF5 file.")

    # Save the file
    with h5py.File(f'{path}/{filename}.h5', 'w') as hdf:
        hdf.create_dataset('dataset', data=image_array)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().detach().numpy()
    min_val = np.min(image_numpy)
    max_val = np.max(image_numpy)

    image_numpy = (image_numpy - min_val) / (max_val - min_val)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0

    return image_numpy.astype(imtype)


def tensor2realseg(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().detach().numpy()
    min_val = np.min(image_numpy)
    max_val = np.max(image_numpy)

    image_numpy = (image_numpy - min_val) / (max_val - min_val)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy[:, :, 1]

    return image_numpy.astype(imtype)


def tensor2im_real(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255

    return image_numpy.astype(imtype)


def tensor2map(image_tensor, imtype=np.uint8):
    if (len(image_tensor.shape) > 3):
        image_numpy = image_tensor[0].cpu().float().numpy()
    else:
        image_numpy = image_tensor.cpu().float().numpy()

    min_val = float(np.min(image_numpy))
    max_val = float(np.max(image_numpy))

    image_numpy = (image_numpy - min_val) / (max_val - min_val)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))

    return image_numpy


def tensor2seg(image_tensor, imtype=np.uint8):
    if (len(image_tensor.shape) > 3):
        image_numpy = image_tensor[0].cpu().float().numpy()
    else:
        image_numpy = image_tensor.cpu().float().numpy()

    min_val = np.min(image_numpy)
    max_val = np.max(image_numpy)

    print(f" seg Minimum value: {min_val}  shape:{np.shape(image_numpy)}")
    print(f"seg Maximum value: {max_val}")
    image_numpy = (image_numpy - min_val) / (max_val - min_val)

    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255
    return image_numpy.astype(imtype)


def thresh2seg(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.float().numpy()
    image_numpy = image_numpy * 100
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

    if (len(image_numpy.shape) > 2):
        image_pil = Image.fromarray(image_numpy[:, :, 0])
    else:
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_map(image_numpy, image_path):

    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('RdYlGn_r')
    cax = ax.imshow(image_numpy, cmap=cmap)
    fig.colorbar(cax)
    plt.savefig(image_path)
    plt.close()



def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


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


def append_data_to_csv(file_path, ordered_dict_data):
    # Check if the file already exists and if it has content
    try:
        with open(file_path, 'r', newline='') as csvfile:
            header_exists = bool(next(csv.reader(csvfile), None))
    except FileNotFoundError:
        header_exists = False

    # Open the file in append mode ('a')
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ordered_dict_data.keys())

        # Write the header if the file is being created or is empty
        if not header_exists:
            writer.writeheader()

        # Write the OrderedDict data
        writer.writerow(ordered_dict_data)