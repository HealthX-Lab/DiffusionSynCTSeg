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
image = (nib.load(path)).get_fdata()
image = np.nan_to_num(image)
image = image.astype(np.float64)
new_image = nib.Nifti1Image(image, np.eye(4))
nib.save(new_image, save_path)