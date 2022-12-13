import nibabel as nib
import numpy as np
import os
import sys


path = sys.argv[1]
image_name = path.split('/')[-1]
pre = 'float_'
image_name = pre + image_name
save_path = os.path.dirname(path)
save_path = os.path.join(save_path, image_name)
print('image_name', image_name, 'save_path',save_path)

image = (nib.load(path)).get_fdata()
print('first image',np.sum(image),np.min(image),np.max(image))
image = np.nan_to_num(image)
print('extract nan',np.sum(image),np.min(image),np.max(image))
image = image.astype(np.float64)
print('convert to np.float64',np.sum(image),np.min(image),np.max(image))
new_image = nib.Nifti1Image(image, np.eye(4))
print('new_image  ',np.sum(new_image),np.min(new_image),np.max(new_image),'*********')
nib.save(new_image, save_path)