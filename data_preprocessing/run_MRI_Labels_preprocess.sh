#!/bin/bash
# In this script, we send all MRIs and their labels in the Neuromorphometric dataset for preprocessing.
# The script iterates through the provided atlas and label files to submit jobs for image processing.



# Prompt the user to enter the base directory path
echo "Please enter the base directory path:"
read BASE_PATH

# Check if the user entered a value
if [ -z "$BASE_PATH" ]; then
  echo "No base directory path entered. Exiting."
  exit 1
fi

# Prompt the user to enter the Singularity image path
echo "Please enter the path to the Singularity image:"
read SINGULARITY_PATH

# Check if the user entered a value
if [ -z "$SINGULARITY_PATH" ]; then
  echo "No Singularity image path entered. Exiting."
  exit 1
fi

# Get atlas and label file paths for both ADNI and OASIS type of data in Neuromorphometrics dataset
atlas_path=( $(find ${BASE_PATH} -type f \( -name "ADNI*.nii" -o -name "*_MR1.nii" -o -name "*_MR2.nii" \) -print))
label_path=( $(find ${BASE_PATH} -type f -name "*seg.nii" -print))

# Ensure the number of atlas and label files match
if [ ${#atlas_path[@]} -ne ${#label_path[@]} ]; then
  echo "The number of atlas files and label files do not match. Exiting."
  exit 1
fi

# Iterate over the atlas and label file paths
for i in $(seq ${#atlas_path[@]}); do
    atlas=${atlas_path[i-1]}
    label=${label_path[i-1]}
    echo "$atlas $label"
    neurogliaSubmit -I  $SINGULARITY_PATH   \
      -j Regular ./MedImageProcessing.sh  -a $atlas -l $label
done



