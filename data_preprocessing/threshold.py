import nibabel as nib
import numpy as np
import os
import sys
# folder path
l_thr = int(sys.argv[1])
u_thr = int(sys.argv[2])
print('*******',l_thr,u_thr)
dir_path ='/home/rtm/data/final_dataset/OASIS3/normal_ct'#'/home/rtm/data/final_dataset/iDB/CT/normal_ct'
save_dir = '/home/rtm/data/final_dataset/OASIS3/window_ct'#'/home/rtm/data/final_dataset/iDB/CT/window_ct/thrmax'
for folder in os.listdir(dir_path):
    path = os.path.join(dir_path, folder)
    print('path : ', path)
    nib_img = nib.load(path)
    image = nib_img.get_fdata()
    np.clip(image, l_thr, u_thr, out=image)
    prefix = 'window_' + str(folder)
    new_image = nib.Nifti1Image(image.astype(np.float64), affine=nib_img.affine,header=nib_img.header)
    save_path = os.path.join(save_dir, prefix)
    print('save_path', save_path)
    nib.save(new_image, save_path)
