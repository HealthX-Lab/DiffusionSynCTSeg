import nibabel as nib
import numpy as np
import os
import sys

'''
This script applies windowing to NIfTI CT images within a specified directory.
It clips the intensity values of the images between a lower and upper threshold
provided as command line arguments, and saves the processed images to a new directory
with a modified prefix.

Instructions:
1. Set the 'data_dir' variable to the correct base path where your CT data is located.
2. The input data must be in the 'final_dataset/normal_ct' subdirectory under 'data_dir'.
3. The processed images will be saved in the 'final_dataset/window_ct' subdirectory under 'data_dir'.
4. Run the script with two arguments: lower and upper threshold values.
'''


def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <lower_threshold> <upper_threshold>")
        sys.exit(1)

    # Get the lower and upper threshold values from command line arguments
    l_thr = int(sys.argv[1])
    u_thr = int(sys.argv[2])

    # Define the base directory path (set this to your actual base path)
    data_dir = 'path-to-data'

    # Define the input and output directory paths
    input_dir = os.path.join(data_dir, 'final_dataset', 'normal_ct')
    output_dir = os.path.join(data_dir, 'final_dataset', 'window_ct')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, 'window_' + filename)

        try:
            # Load the NIfTI image
            nib_img = nib.load(input_path)
            image = nib_img.get_fdata()

            # Apply windowing
            np.clip(image, l_thr, u_thr, out=image)

            # Create a new NIfTI image
            new_image = nib.Nifti1Image(image.astype(np.float64), affine=nib_img.affine, header=nib_img.header)

            # Save the processed image
            nib.save(new_image, output_path)
            print(f"Processed and saved: {output_path}")

        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    main()
