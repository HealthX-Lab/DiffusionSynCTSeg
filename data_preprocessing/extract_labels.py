import nibabel as nib
import numpy as np
import os

# folder path
dir_path = '/home/rtm/data/final_dataset/iDB/label/beast'
save_dir = '/home/rtm/data/final_dataset/iDB/label/beast_ventricle'


for folder in os.listdir(dir_path):
    # check if current path is a file
    path = os.path.join(dir_path, folder,'image','Labels.nii.gz')
    # path = os.path.join(dir_path, folder,str(folder)+ '_seg.nii')
    nib_img = nib.load(path)
    image = nib_img.get_fdata()
    arr4 = np.where(image == 4, 50, 0)
    arr11 = np.where(image == 11, 100, 0)
    arr15 = np.where(image == 15, 150, 0)
    arr52 = np.where(image == 52, 200, 0)
    arr51 = np.where(image == 51, 250, 0)
    arr = arr4 + arr11 + arr15 + arr52 + arr51


    prefix = 'label_' + str(folder) + '.nii.gz'
    new_image = nib.Nifti1Image(arr.astype(np.float64), affine=nib_img.affine,header=nib_img.header)
    # save_path = os.path.join(save_dir, folder+'.nii.gz')
    save_path = os.path.join(save_dir, prefix)
    print('save_path', save_path)
    nib.save(new_image, save_path)
