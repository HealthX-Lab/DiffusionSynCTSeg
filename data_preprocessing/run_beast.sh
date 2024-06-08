#!/bin/bash

# This script runs BEaST for MRI images to extract the brain and delete the skull.

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
mri_path=( $(ls -a ${BASE_PATH}/*.nii.gz))

# Iterate over the MRI image paths
for i in $(seq ${#mri_path[@]}); do
    mri=${mri_path[i-1]}
    mri_input_basename="$(basename -- $mri)"
    echo 'Processing MRI: ' $mri_input_basename

    # Submit the job using neurogliaSubmit with the specified Singularity image and MRI path
    neurogliaSubmit -I $SINGULARITY_PATH -j 2core8gb ./MedImageProcessing.sh -b $mri
done

echo "Processing complete."