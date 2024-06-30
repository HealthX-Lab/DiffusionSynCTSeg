#!/bin/bash

# This script preprocesses MRI images using a specified Singularity image and a list of MRI image paths.
# Prompt the user to enter the base directory path containing the MRI images

echo "Please enter the base directory path containing the MRI images:"
read BASE_PATH

# Check if the user entered a value
if [ -z "$BASE_PATH" ]; then
  echo "No base directory path entered. Exiting."
  exit 1
fi

# Prompt the user to enter the path to the Singularity image
echo "Please enter the path to the Singularity image:"
read SINGULARITY_PATH

# Check if the user entered a value
if [ -z "$SINGULARITY_PATH" ]; then
  echo "No Singularity image path entered. Exiting."
  exit 1
fi


# Get the list of MRI image paths
target_mri_path=( $(ls -a ${BASE_PATH}/*mri.nii.gz))

# Iterate over the MRI image paths
for i in $(seq ${#target_mri_path[@]}); do
    target_mri=${target_mri_path[i-1]}

    echo "Processing MRI image: $target_mri"
    neurogliaSubmit -I $SINGULARITY_PATH -j Regular ./MedImageProcessing.sh -w $target_mri
done

echo "Processing complete."