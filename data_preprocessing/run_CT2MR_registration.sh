#!/bin/bash

#!/bin/bash

# This script registers CT images to MRI in the test dataset using a specified Singularity image.
# Prompt the user to enter the base directory path containing the MRI images
echo "Please enter the base directory path containing the MRI images:"
read BASE_MRI_PATH

# Check if the user entered a value
if [ -z "$BASE_MRI_PATH" ]; then
  echo "No base directory path entered. Exiting."
  exit 1
fi

# Prompt the user to enter the base directory path containing the CT images
echo "Please enter the base directory path containing the CT images:"
read BASE_CT_PATH

# Check if the user entered a value
if [ -z "$BASE_CT_PATH" ]; then
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

# Get the list of MRI and CT image paths
MRI_path=( $(ls -a ${BASE_MRI_PATH}/*T1w.nii.gz))
CT_path=( $(ls -a ${BASE_CT_PATH}/*ct.nii.gz))

# Ensure the number of MRI and CT images match
if [ ${#MRI_path[@]} -ne ${#CT_path[@]} ]; then
  echo "The number of MRI and CT images do not match. Exiting."
  exit 1
fi

# Iterate over the MRI and CT image paths
for i in $(seq ${#MRI_path[@]}); do
    MRI=${MRI_path[i-1]}
    CT=${CT_path[i-1]}
    echo 'Processing MRI and CT paths: '$MRI $CT '**********'

    # Submit the job using neurogliaSubmit with the specified Singularity image and paths
    neurogliaSubmit -I $SINGULARITY_PATH -j Quick ./MedImageProcessing.sh -m $MRI -r $CT
done

echo "Processing complete."