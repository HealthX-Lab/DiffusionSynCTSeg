#!/bin/bash

# Setting the ANTSPATH environment variable
export ANTSPATH=/opt/minc/1.9.16/bin/

# Initialize input_atlas_labels_name
input_atlas_labels_name=" "

# Prompt the user to enter the required paths
echo "Please enter the path to the atlas label directory:"
read atlas_label_path
echo "Please enter the path to the skull stripped atlas directory:"
read skull_stripped_atlas_path

# Check if the user entered a value for atlas_label_path
if [ -z "$atlas_label_path" ]; then
  echo "No atlas label directory path entered. Exiting."
  exit 1
fi

# Check if the user entered a value for skull_stripped_atlas_path
if [ -z "$skull_stripped_atlas_path" ]; then
  echo "No skull stripped atlas directory path entered. Exiting."
  exit 1
fi



# Prompt the user to enter the required paths
echo "Please enter the path to the target Skull Stripped images directory:"
read skull_stripped_images_path
echo "Please enter the path to the save directory:"
read save_directory_path

# Check if the user entered a value for skull_stripped_images_path
if [ -z "$skull_stripped_images_path" ]; then
  echo "No Skull Stripped images directory path entered. Exiting."
  exit 1
fi

# Check if the user entered a value for save_directory_path
if [ -z "$save_directory_path" ]; then
  echo "No save directory path entered. Exiting."
  exit 1
fi

echo "Please enter the path to the Singularity container:"
read singularity_path


# Check if the user entered a value for singularity_path
if [ -z "$singularity_path" ]; then
  echo "No Singularity container path entered. Exiting."
  exit 1
fi


# Get the paths for the label images
label_path=( $(ls -a ${atlas_label_path}/*seg.nii.gz))

# Enable extended pattern matching features
shopt -s extglob

# Get the paths for the atlas images excluding files with 'mask.nii.gz' in their names
atlas_path=($(ls -a ${skull_stripped_atlas_path}/!(*mask.nii.gz)))


# get all atlases with their labels
for i in $(seq ${#atlas_path[@]}); do
    atlas=${atlas_path[i-1]}
    label=${label_path[i-1]}
    echo $atlas
    echo $label
    input_atlas_labels_name+=" -g $atlas -l $label"
done


# Get the paths for the target images
target_path=( $(ls -a ${skull_stripped_images_path}/*T1w.nii.gz))
echo $target_path


# Loop through each target image path to create label for each target mri
for i in $(seq ${#trget_path[@]}); do
  # Get the current target image path
  target=${trget_path[i-1]}
  echo 'target***'  $target

  # Get the basename of the target image
  target_basename="$(basename -- $target)"
  echo 'target_basename' $target_basename

  # Remove the postfix from the target image basename
  target_basename_without_postfix=$(echo $target_basename | cut -d . -f 1 -)
  echo 'target_basename_without_postfix' $target_basename_without_postfix

  # Define the output path
  image_folder='/image/'
  output_path="$save_dir$target_basename_without_postfix$image_folder"
  echo 'output_path' $output_path

  mkdir -p $output_path
  # Submit the job to neuroglia with the Singularity container
  neurogliaSubmit -I $singularity_path -j Regular ./antsJointLabelFusion.sh -d 3 -t $target -o $output_path -p malfPosteriors%04d.nii.gz $input_atlas_labels_name
  done
