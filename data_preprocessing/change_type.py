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
nib_img = nib.load(path)
image = nib_img.get_fdata()
image = np.nan_to_num(image)
image = image.astype(np.float64)
new_image = nib.Nifti1Image(image,  affine=nib_img.affine,header=nib_img.header)
nib.save(new_image, save_path)