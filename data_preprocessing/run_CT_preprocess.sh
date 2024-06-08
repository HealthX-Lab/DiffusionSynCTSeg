#!/bin/bash

# This script preprocesses CT images using a specified Singularity image and a list of CT image paths.
# Prompt the user to enter the base directory path containing the CT images

echo "Please enter the base directory path containing the CT images:"
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


# Get the list of CT image paths
target_ct_path=( $(ls -a ${BASE_PATH}/*ct.nii.gz))

# Iterate over the CT image paths
for i in $(seq ${#target_ct_path[@]}); do
    target_ct=${target_ct_path[i-1]}

    echo "Processing CT image: $target_ct"
    neurogliaSubmit -I $SINGULARITY_PATH -j Regular ./MedImageProcessing.sh -c $target_ct
done

echo "Processing complete."