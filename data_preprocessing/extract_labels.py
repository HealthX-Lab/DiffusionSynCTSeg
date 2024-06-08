import nibabel as nib
import numpy as np
import os

'''
This script extracts ventricle labels from whole brain NIfTI images.
It loads labeled images, extracts specific ventricle labels, and saves the processed images to a specified directory.

Instructions:
1. The script will prompt you to enter the directory path containing the labeled images.
2. The script saves the extracted ventricle images one directory above the input directory in a 'ventricle_label' folder.
3. The script processes the following ventricle labels based on the given XML:
   - 3rd Ventricle (4)
   - 4th Ventricle (11)
   - 5th Ventricle (15)
   - Left Lateral Ventricle (52)
   - Right Lateral Ventricle (51)
   
Input Directory Structure:
Each image has its own folder within the input directory, and each folder contains a 'Labels.nii.gz' file in an 'image' subfolder.

'''


# Prompt the user to enter the directory path containing the labeled images
dir_path = input("Please enter the directory path containing the labeled images: ")

# Determine the parent directory of dir_path and create the save directory path
parent_dir = os.path.dirname(dir_path)
save_dir = os.path.join(parent_dir, 'ventricle_label')

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Process each folder in the input directory
for folder in os.listdir(dir_path):
    # Construct the path to the label image
    path = os.path.join(dir_path, folder, 'Labels.nii.gz')

    try:
        # Load the NIfTI image
        nib_img = nib.load(path)
        image = nib_img.get_fdata()

        # Extract ventricle labels and assign new values
        # 3rd Ventricle (4)
        arr4 = np.where(image == 4, 50, 0)
        # 4th Ventricle (11)
        arr11 = np.where(image == 11, 100, 0)
        # 5th Ventricle (15)
        arr15 = np.where(image == 15, 150, 0)
        # Left Lateral Ventricle (52)
        arr52 = np.where(image == 52, 200, 0)
        # Right Lateral Ventricle (51)
        arr51 = np.where(image == 51, 250, 0)
        arr = arr4 + arr11 + arr15 + arr52 + arr51

        # Create a new NIfTI image
        prefix = 'label_' + str(folder) + '.nii.gz'
        new_image = nib.Nifti1Image(arr.astype(np.float64), affine=nib_img.affine, header=nib_img.header)

        # Construct the save path and save the processed image
        save_path = os.path.join(save_dir, prefix)
        print('Saving to:', save_path)
        nib.save(new_image, save_path)

    except Exception as e:
        print(f"Error processing {path}: {e}")
