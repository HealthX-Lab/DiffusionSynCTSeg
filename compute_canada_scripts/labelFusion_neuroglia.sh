#!/bin/bash

export ANTSPATH=/opt/minc/1.9.16/bin/



input_atlas_labels_name=" "

BASE_PATH=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/atlas_label
label_path=( $( ls -a ${BASE_PATH}/*seg.nii.gz))
BASE_PATH=/home/rtm/scratch/rtm/data/MedImagepreprocess/SkullStrippedimages/atlas
shopt -s extglob
atlas_path=($( ls -a ${BASE_PATH}/!(*mask.nii.gz)))



for i in $(seq ${#atlas_path[@]}); do
    atlas=${atlas_path[i-1]}
    label=${label_path[i-1]}
    echo $atlas
    echo $label
    echo "*************"

    input_atlas_labels_name+=" -g $atlas -l $label"

done

echo $input_atlas_labels_name






BASE_PATH=/home/rtm/scratch/rtm/data/MedImagepreprocess/SkullStrippedimages/iDB_mri
trget_path=( $( ls -a ${BASE_PATH}/*T1w.nii.gz))
save_dir=/home/rtm/scratch/rtm/data/MedImagepreprocess/labelfiusion/
echo $target_path

for i in $(seq ${#trget_path[@]}); do
    target=${trget_path[i-1]}
    echo 'target***'  $target



    target_basename="$(basename -- $target)"
  echo 'target_basename' $target_basename #sub-0001_space-pet_FLAIR.nii.gz
  target_basename_without_postfix=$(echo $target_basename | cut -d . -f 1 -)
  echo 'target_basename_without_postfix' $target_basename_without_postfix  #sub-0001_space-pet_FLAIR
  image_folder='/image/'
  output_path="$save_dir$target_basename_without_postfix$image_folder"
  echo 'output_path' $output_path #/home/rtm/scratch/rtm/data/labelFusion/result/sub-0001_space-pet_FLAIR/

  mkdir -p $output_path



    neurogliaSubmit -I /project/6055004/tools/singularity/minc-toolkit-1.9.16-min.simg -j Regular /scratch/rtm/rtm/ms_project/domain_adaptation_CTscan/scripts/antsJointLabelFusion.sh -d 3 -t $target -o $output_path -p malfPosteriors%04d.nii.gz $input_atlas_labels_name


  done
#
